#![forbid(unsafe_code)]
//! A storage node.
//!
//! A node owns a local `TensorCache` and participates in the distributed
//! runtime by registering with the coordinator, serving peer fetches, storing
//! replicas, and participating in coordinator-driven migration. The node is
//! run as an independent OS process; its TCP server (peer + client protocol)
//! is wired up by the CLI, while this module holds the protocol logic and the
//! coordinator/peer interactions.

use std::net::TcpStream;
use std::time::Duration;

use crate::compat::CompatKey;
use crate::crc::crc32c;
use crate::error::{Error, Result};
use crate::ident::{Address, ObjectId};
use crate::protocol::{read_frame, write_frame, Message};
use crate::runtime::TensorCache;
use crate::tiers::Tier;

/// A storage node identity and coordinator binding.
pub struct Node {
    pub node_id: String,
    pub node_addr: String,
    pub coordinator_addr: String,
    pub tc: TensorCache,
    pub lease_ns: u64,
}

impl Node {
    pub fn new(
        node_id: String,
        node_addr: String,
        coordinator_addr: String,
        tc: TensorCache,
        lease_ns: u64,
    ) -> Self {
        Node {
            node_id,
            node_addr,
            coordinator_addr,
            tc,
            lease_ns,
        }
    }

    fn connect(&self, addr: &str) -> Result<TcpStream> {
        let socket: std::net::SocketAddr = addr
            .parse::<std::net::SocketAddr>()
            .map_err(|e| Error::Io(e.to_string()))?;
        TcpStream::connect_timeout(&socket, Duration::from_secs(30))
            .map_err(|e| Error::Io(e.to_string()))
    }

    /// A one-request/one-response exchange with the coordinator.
    fn coordinator_roundtrip(&self, req: &Message) -> Result<Message> {
        let mut stream = self.connect(&self.coordinator_addr)?;
        write_frame(&mut stream, req)?;
        let (t, payload) = read_frame(&mut stream)?
            .ok_or_else(|| Error::Protocol("coordinator closed connection".into()))?;
        Message::decode(t, &payload)
    }

    /// Register with the coordinator and receive the authority Hello.
    pub fn register(&self) -> Result<Message> {
        self.coordinator_roundtrip(&Message::Register {
            node_id: self.node_id.clone(),
            addr: self.node_addr.clone(),
        })
    }

    /// Handle an inbound message on the peer/client server, returning responses.
    /// `owner_fence` is the last fence the node has observed for any owned
    /// object (used for migration requests).
    pub fn handle_peer(&self, msg: &Message) -> Result<Vec<Message>> {
        match msg {
            Message::Register { .. } => Err(Error::Protocol("nodes do not serve Register".into())),
            Message::Store {
                namespace,
                key,
                generation,
                data,
                crc,
                compat,
                source,
            } => {
                let expected = crc32c(data);
                if *crc != expected {
                    return Err(Error::Integrity("Store payload CRC mismatch".into()));
                }
                let compat_key = CompatKey::decode(compat)?;
                let oid = Address::new(namespace.clone(), key.clone(), *generation).object_id();
                match self
                    .tc
                    .register(namespace, key, *generation, compat_key.clone(), data)
                {
                    Ok(_) => {}
                    Err(Error::Exists(_)) => {
                        // Idempotent replica store: the new owner may already hold
                        // a replica. Verify compatibility and byte-equality.
                        let stored = self.tc.entry_compat_id(&oid)?;
                        if stored != compat_key.compat_id() {
                            return Err(Error::Compatibility(
                                "store conflicts with an existing incompatible object".into(),
                            ));
                        }
                        let existing = self.tc.restore(&oid, &Tier::Host)?;
                        if existing != *data {
                            return Err(Error::Integrity(
                                "store payload differs from the existing object".into(),
                            ));
                        }
                    }
                    Err(e) => return Err(e),
                }
                // If this object is not yet owned, the first writer becomes owner.
                let lk = self.coordinator_roundtrip(&Message::Lookup {
                    namespace: namespace.clone(),
                    key: key.clone(),
                    generation: *generation,
                    compat: compat.clone(),
                })?;
                if let Message::LookupResult { found: false, .. } = lk {
                    let _ = self.coordinator_roundtrip(&Message::Create {
                        namespace: namespace.clone(),
                        key: key.clone(),
                        generation: *generation,
                        byte_len: data.len() as u64,
                        compat: compat.clone(),
                        node_id: self.node_id.clone(),
                    })?;
                }
                let _ = source;
                Ok(vec![Message::StoreAck {
                    object_id: oid.to_hex(),
                }])
            }
            Message::Fetch { object_id, compat } => {
                let oid = ObjectId::from_hex(object_id)?;
                let stored = self.tc.entry_compat_id(&oid)?;
                let req_compat = CompatKey::decode(compat)?;
                if stored != req_compat.compat_id() {
                    return Err(Error::Compatibility(format!(
                        "object {oid} is not compatible with request"
                    )));
                }
                // Only serve if the object is actually present.
                let bytes = self.tc.restore(&oid, &Tier::Host)?;
                let crc = crc32c(&bytes);
                Ok(vec![Message::FetchReply {
                    object_id: object_id.clone(),
                    data: bytes,
                    crc,
                }])
            }
            Message::Lookup {
                namespace,
                key,
                generation,
                compat,
            } => {
                let lk = self.coordinator_roundtrip(&Message::Lookup {
                    namespace: namespace.clone(),
                    key: key.clone(),
                    generation: *generation,
                    compat: compat.clone(),
                })?;
                match lk {
                    Message::LookupResult {
                        found: true,
                        owner,
                        owner_addr,
                        generation,
                    } => {
                        if owner.as_deref() == Some(self.node_id.as_str()) {
                            Ok(vec![Message::LookupResult {
                                found: true,
                                owner,
                                owner_addr: Some(self.node_addr.clone()),
                                generation,
                            }])
                        } else if let Some(addr) = owner_addr {
                            // Fetch from the owner and store a local replica.
                            let oid_hex = Address::new(namespace.clone(), key.clone(), generation)
                                .object_id()
                                .to_hex();
                            let data = self.fetch_from(&addr, &oid_hex, compat)?;
                            let compat_key = CompatKey::decode(compat)?;
                            let _ = self
                                .tc
                                .register(namespace, key, generation, compat_key, &data);
                            Ok(vec![Message::LookupResult {
                                found: true,
                                owner,
                                owner_addr: Some(addr),
                                generation,
                            }])
                        } else {
                            Ok(vec![Message::LookupResult {
                                found: true,
                                owner,
                                owner_addr: None,
                                generation,
                            }])
                        }
                    }
                    Message::LookupResult {
                        found: false,
                        generation,
                        ..
                    } => Ok(vec![Message::LookupResult {
                        found: false,
                        owner: None,
                        owner_addr: None,
                        generation,
                    }]),
                    other => Err(Error::Protocol(format!(
                        "unexpected coordinator reply {:?}",
                        other.msg_type()
                    ))),
                }
            }
            Message::CreateAck {
                object_id,
                epoch,
                fence,
                owner,
            } => {
                // A client result; used internally. Expose as-is for the caller.
                Ok(vec![Message::CreateAck {
                    object_id: object_id.clone(),
                    epoch: *epoch,
                    fence: *fence,
                    owner: owner.clone(),
                }])
            }
            Message::Migrate {
                object_id,
                new_owner,
                new_owner_addr,
                fence,
            } => {
                self.migrate(object_id, new_owner, new_owner_addr, *fence)?;
                Ok(vec![Message::MigrateAck {
                    object_id: object_id.clone(),
                    new_owner: new_owner.clone(),
                    fence: *fence,
                }])
            }
            _ => Ok(vec![Message::Error {
                code: "unsupported".into(),
                message: format!("node does not handle {}", msg.msg_type().tag()),
            }]),
        }
    }

    /// Fetch an object's bytes from a peer node (owner).
    pub fn fetch_from(&self, owner_addr: &str, object_id: &str, compat: &[u8]) -> Result<Vec<u8>> {
        let mut stream = self.connect(owner_addr)?;
        write_frame(
            &mut stream,
            &Message::Fetch {
                object_id: object_id.to_string(),
                compat: compat.to_vec(),
            },
        )?;
        let (t, payload) = read_frame(&mut stream)?
            .ok_or_else(|| Error::Protocol("owner closed connection".into()))?;
        match Message::decode(t, &payload)? {
            Message::FetchReply { data, crc, .. } => {
                if crc32c(&data) != crc {
                    return Err(Error::Integrity("fetch payload CRC mismatch".into()));
                }
                Ok(data)
            }
            Message::Error { message, .. } => Err(Error::Protocol(message)),
            other => Err(Error::Protocol(format!(
                "unexpected fetch reply {:?}",
                other.msg_type()
            ))),
        }
    }

    /// Push an object to a peer node (used during migration).
    pub fn store_to(
        &self,
        peer_addr: &str,
        namespace: &str,
        key: &str,
        generation: u64,
        compat: &[u8],
        data: &[u8],
    ) -> Result<()> {
        let mut stream = self.connect(peer_addr)?;
        let crc = crc32c(data);
        write_frame(
            &mut stream,
            &Message::Store {
                namespace: namespace.to_string(),
                key: key.to_string(),
                generation,
                data: data.to_vec(),
                crc,
                compat: compat.to_vec(),
                source: self.node_id.clone(),
            },
        )?;
        let (t, payload) = read_frame(&mut stream)?
            .ok_or_else(|| Error::Protocol("peer closed connection".into()))?;
        match Message::decode(t, &payload)? {
            Message::StoreAck { .. } => Ok(()),
            Message::Error { message, .. } => Err(Error::Protocol(format!(
                "store rejected by peer: {message}"
            ))),
            other => Err(Error::Protocol(format!(
                "unexpected store reply {:?}",
                other.msg_type()
            ))),
        }
    }

    /// Drive a coordinator-authorized migration of `object_id` to `new_owner`.
    pub fn migrate(
        &self,
        object_id: &str,
        new_owner: &str,
        new_owner_addr: &str,
        fence: u64,
    ) -> Result<()> {
        let mut stream = self.connect(&self.coordinator_addr)?;
        write_frame(
            &mut stream,
            &Message::Migrate {
                object_id: object_id.to_string(),
                new_owner: new_owner.to_string(),
                new_owner_addr: new_owner_addr.to_string(),
                fence,
            },
        )?;
        let (t, payload) = read_frame(&mut stream)?
            .ok_or_else(|| Error::Protocol("coordinator closed during migration".into()))?;
        let instr = Message::decode(t, &payload)?;
        match instr {
            Message::Migrate {
                object_id,
                new_owner,
                fence,
                ..
            } => {
                // Read the local object and push it to the new owner.
                let oid = ObjectId::from_hex(&object_id)?;
                let meta = self.tc.metadata(&oid)?;
                let compat = self.tc.entry_compat(&oid)?;
                let data = self.tc.restore(&oid, &Tier::Host)?;
                self.store_to(
                    new_owner_addr,
                    &meta.namespace,
                    &meta.key,
                    meta.generation,
                    &compat.encode(),
                    &data,
                )?;
                // Acknowledge to the coordinator on the same connection.
                write_frame(
                    &mut stream,
                    &Message::MigrateAck {
                        object_id,
                        new_owner,
                        fence,
                    },
                )?;
                Ok(())
            }
            Message::Error { message, .. } => Err(Error::Protocol(message)),
            other => Err(Error::Protocol(format!(
                "unexpected migrate reply {:?}",
                other.msg_type()
            ))),
        }
    }
}

/// Parse a "node" reference: a node id and a peer address, e.g. "n1@127.0.0.1:9001".
pub fn parse_node_ref(s: &str) -> Result<(String, String)> {
    let (id, addr) = s
        .split_once('@')
        .ok_or_else(|| Error::InvalidArgument("node reference must be id@addr".into()))?;
    Ok((id.to_string(), addr.to_string()))
}

// Re-export MsgType for convenience (used by the CLI).
pub use crate::protocol::MsgType;
