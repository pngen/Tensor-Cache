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
                new_owner_addr,
                fence,
            } => {
                // Read the local object and push it to the new owner, at the
                // address the coordinator names for it. The address passed in
                // by the requester is only what the coordinator falls back to
                // for a node it does not know.
                let oid = ObjectId::from_hex(&object_id)?;
                let meta = self.tc.metadata(&oid)?;
                let compat = self.tc.entry_compat(&oid)?;
                let data = self.tc.restore(&oid, &Tier::Host)?;
                self.store_to(
                    &new_owner_addr,
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
                // Ownership has moved only once the coordinator has taken the
                // acknowledgement. It answers one it rejects with an error and
                // otherwise just finishes the connection, so wait for that
                // before reporting the migration as done.
                stream
                    .shutdown(std::net::Shutdown::Write)
                    .map_err(|e| Error::Io(e.to_string()))?;
                match read_frame(&mut stream)? {
                    None => Ok(()),
                    Some((t, payload)) => match Message::decode(t, &payload)? {
                        Message::Error { message, .. } => Err(Error::Migration(format!(
                            "coordinator rejected the migration: {message}"
                        ))),
                        other => Err(Error::Protocol(format!(
                            "unexpected migrate-ack reply {:?}",
                            other.msg_type()
                        ))),
                    },
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::geometry::{Layout, Shape};
    use crate::runtime::RuntimeConfig;
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    fn listener() -> (TcpListener, String) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        (l, addr)
    }

    /// A node that holds one object and talks to `coordinator_addr`.
    fn node_with_object(coordinator_addr: String) -> (Node, String) {
        let tc = TensorCache::new(RuntimeConfig {
            host_capacity: 1 << 20,
            ..Default::default()
        })
        .unwrap();
        let compat = CompatKey {
            dtype: Dtype::F32,
            shape: Shape::new(vec![16, 16]).unwrap(),
            layout: Layout::RowMajor,
            model: Some("node-test".into()),
            ..Default::default()
        };
        let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let oid = tc.register("ns", "k", 1, compat, &data).unwrap();
        let node = Node::new(
            "n1".into(),
            "127.0.0.1:1".into(),
            coordinator_addr,
            tc,
            1_000_000_000,
        );
        (node, oid.to_hex())
    }

    /// A peer that acknowledges every Store it is sent and reports how many
    /// it received. It stops at the first connection that sends nothing.
    fn spawn_peer(listener: TcpListener) -> JoinHandle<usize> {
        std::thread::spawn(move || {
            let mut stores = 0;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let Ok(Some((t, payload))) = read_frame(&mut stream) else {
                    break;
                };
                if let Ok(Message::Store {
                    namespace,
                    key,
                    generation,
                    ..
                }) = Message::decode(t, &payload)
                {
                    stores += 1;
                    let object_id = Address::new(namespace, key, generation)
                        .object_id()
                        .to_hex();
                    let _ = write_frame(&mut stream, &Message::StoreAck { object_id });
                }
            }
            stores
        })
    }

    fn stop_peer(addr: &str, peer: JoinHandle<usize>) -> usize {
        // A connection that sends nothing ends the peer's accept loop.
        drop(TcpStream::connect(addr));
        peer.join().unwrap()
    }

    /// A coordinator that answers one Migrate request with an instruction for
    /// `target_addr`, then takes the node's acknowledgement and either accepts
    /// it (finishes the connection) or rejects it with an error. Reports
    /// whether an acknowledgement arrived.
    fn spawn_coordinator(
        listener: TcpListener,
        target_addr: String,
        reject_ack: bool,
    ) -> JoinHandle<bool> {
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return false;
            };
            let Ok(Some((t, payload))) = read_frame(&mut stream) else {
                return false;
            };
            let Ok(Message::Migrate {
                object_id,
                new_owner,
                fence,
                ..
            }) = Message::decode(t, &payload)
            else {
                return false;
            };
            let _ = write_frame(
                &mut stream,
                &Message::Migrate {
                    object_id,
                    new_owner,
                    new_owner_addr: target_addr,
                    fence: fence + 1,
                },
            );
            let acked = matches!(read_frame(&mut stream), Ok(Some((MsgType::MigrateAck, _))));
            if acked && reject_ack {
                let _ = write_frame(
                    &mut stream,
                    &Message::Error {
                        code: "authority".into(),
                        message: "migration ack mismatch".into(),
                    },
                );
            }
            acked
        })
    }

    #[test]
    fn migration_pushes_the_object_to_the_address_the_coordinator_names() {
        let (coord_listener, coord_addr) = listener();
        // Where the coordinator says the new owner is, and what the requester
        // claimed instead.
        let (owner_listener, owner_addr) = listener();
        let (other_listener, other_addr) = listener();
        let coordinator = spawn_coordinator(coord_listener, owner_addr.clone(), false);
        let owner = spawn_peer(owner_listener);
        let other = spawn_peer(other_listener);
        let (node, oid) = node_with_object(coord_addr);

        let result = node.migrate(&oid, "n2", &other_addr, 0);
        let acked = coordinator.join().unwrap();
        let at_owner = stop_peer(&owner_addr, owner);
        let at_other = stop_peer(&other_addr, other);

        result.unwrap();
        assert!(acked);
        assert_eq!(
            (at_owner, at_other),
            (1, 0),
            "the object must reach the coordinator's target, and only it"
        );
    }

    #[test]
    fn migration_fails_when_the_coordinator_rejects_the_acknowledgement() {
        let (coord_listener, coord_addr) = listener();
        let (owner_listener, owner_addr) = listener();
        let coordinator = spawn_coordinator(coord_listener, owner_addr.clone(), true);
        let owner = spawn_peer(owner_listener);
        let (node, oid) = node_with_object(coord_addr);

        let result = node.migrate(&oid, "n2", &owner_addr, 0);
        let acked = coordinator.join().unwrap();
        let stored = stop_peer(&owner_addr, owner);

        assert!(acked, "the acknowledgement reaches the coordinator");
        assert_eq!(stored, 1);
        assert!(
            matches!(result, Err(Error::Migration(_))),
            "ownership did not move, so the migration must not succeed: {result:?}"
        );
    }
}
