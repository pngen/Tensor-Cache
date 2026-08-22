#![forbid(unsafe_code)]
//! The coordinator: distributed authority, ownership and replica tracking.
//!
//! The coordinator is a stateless-as-possible registry that is the single
//! source of authority. It holds a monotonic epoch and an immutable boot
//! identity, tracks registered nodes and their peer addresses, maps objects to
//! their authoritative owner, and issues ownership fences. A coordinator
//! restart advances the epoch and generates a new boot identity, which
//! invalidates every node's prior lease.
//!
//! This module is pure logic; network plumbing lives in the CLI's coordinator
//! command. `handle` maps an inbound message to zero or more outbound messages.

use std::collections::HashMap;
use std::path::Path;

use crate::authority::{BootId, Epoch};
use crate::error::{Error, Result};
use crate::ident::Address;
use crate::protocol::Message;
use crate::wire::{Reader, Writer};

/// The authoritative owner and fence for one object.
#[derive(Debug, Clone)]
pub struct ObjectState {
    pub owner: String,
    pub fence: u64,
    pub epoch: Epoch,
}

/// A migration in progress.
#[derive(Debug, Clone)]
pub struct PendingMigration {
    pub new_owner: String,
    pub to_fence: u64,
}

/// The coordinator state.
#[derive(Debug)]
pub struct Coordinator {
    epoch: Epoch,
    boot_id: BootId,
    lease_ns: u64,
    nodes: HashMap<String, String>,
    objects: HashMap<String, ObjectState>,
    migrations: HashMap<String, PendingMigration>,
}

impl Coordinator {
    pub fn new(lease_ns: u64) -> Self {
        Coordinator {
            epoch: Epoch::new(1),
            boot_id: BootId::new(),
            lease_ns,
            nodes: HashMap::new(),
            objects: HashMap::new(),
            migrations: HashMap::new(),
        }
    }

    /// Simulate a coordinator restart: advance epoch and rotate boot identity,
    /// which invalidates all prior node leases.
    pub fn restart(&mut self) {
        self.epoch = self.epoch.next();
        self.boot_id = BootId::new();
    }

    /// Serialize the durable authority/ownership snapshot.
    pub fn save_state(&self, path: &Path) -> Result<()> {
        let mut w = Writer::new();
        w.u32(0x5443_434F); // "TCCO"
        w.u8(1);
        w.u64(self.epoch.value());
        w.u64(self.objects.len() as u64);
        for (id, st) in &self.objects {
            w.str(id);
            w.str(&st.owner);
            w.u64(st.fence);
        }
        let bytes = w.into_inner();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a snapshot, if present, and advance the epoch/rotate boot identity
    /// to reflect a restart. Preserves object ownership so a coordinator
    /// restart does not orphan existing objects.
    pub fn load_state(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }
        let bytes = std::fs::read(path)?;
        let mut r = Reader::new(&bytes)?;
        if r.u32()? != 0x5443_434F {
            return Err(Error::Persistence("bad coordinator snapshot magic".into()));
        }
        let version = r.u8()?;
        if version != 1 {
            return Err(Error::Persistence(
                "unsupported coordinator snapshot version".into(),
            ));
        }
        let epoch = r.u64()?;
        let count = r.u64()?;
        let mut objects = HashMap::new();
        for _ in 0..count {
            let id = r.str()?.to_owned();
            let owner = r.str()?.to_owned();
            let fence = r.u64()?;
            objects.insert(
                id,
                ObjectState {
                    owner,
                    fence,
                    epoch: Epoch::new(epoch),
                },
            );
        }
        if !r.eof() {
            return Err(Error::Persistence(
                "trailing bytes in coordinator snapshot".into(),
            ));
        }
        self.objects = objects;
        // Restart semantics: advance epoch and rotate boot identity.
        self.epoch = Epoch::new(epoch.saturating_add(1));
        self.boot_id = BootId::new();
        Ok(())
    }

    pub fn epoch(&self) -> u64 {
        self.epoch.value()
    }

    pub fn boot_id(&self) -> &BootId {
        &self.boot_id
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    fn object_id(namespace: &str, key: &str, generation: u64) -> String {
        Address::new(namespace.to_string(), key.to_string(), generation)
            .object_id()
            .to_hex()
    }

    /// Handle one inbound message, returning any responses to write back to the
    /// sender.
    pub fn handle(&mut self, msg: &Message) -> Result<Vec<Message>> {
        match msg {
            Message::Register { node_id, addr } => {
                self.nodes.insert(node_id.clone(), addr.clone());
                Ok(vec![Message::Hello {
                    epoch: self.epoch.value(),
                    boot_id: self.boot_id.as_str().to_string(),
                    node_id: node_id.clone(),
                    addr: addr.clone(),
                    lease_ns: self.lease_ns,
                }])
            }
            Message::Create {
                namespace,
                key,
                generation,
                node_id,
                ..
            } => {
                let id = Self::object_id(namespace, key, *generation);
                if let Some(state) = self.objects.get_mut(&id) {
                    // Existing object: allow idempotent re-claim by the current
                    // owner with a matching fence; otherwise reject as stale.
                    if state.owner == *node_id {
                        return Ok(vec![Message::CreateAck {
                            object_id: id,
                            epoch: self.epoch.value(),
                            fence: state.fence,
                            owner: node_id.clone(),
                        }]);
                    }
                    return Err(Error::Authority(format!(
                        "object {id} already owned by {}",
                        state.owner
                    )));
                }
                self.objects.insert(
                    id.clone(),
                    ObjectState {
                        owner: node_id.clone(),
                        fence: 0,
                        epoch: self.epoch,
                    },
                );
                Ok(vec![Message::CreateAck {
                    object_id: id,
                    epoch: self.epoch.value(),
                    fence: 0,
                    owner: node_id.clone(),
                }])
            }
            Message::Lookup {
                namespace,
                key,
                generation,
                ..
            } => {
                let id = Self::object_id(namespace, key, *generation);
                match self.objects.get(&id) {
                    Some(state) => Ok(vec![Message::LookupResult {
                        found: true,
                        owner: Some(state.owner.clone()),
                        owner_addr: self.nodes.get(&state.owner).cloned(),
                        generation: *generation,
                    }]),
                    None => Ok(vec![Message::LookupResult {
                        found: false,
                        owner: None,
                        owner_addr: None,
                        generation: *generation,
                    }]),
                }
            }
            Message::LeaseRenew { object_id, fence } => match self.objects.get(object_id) {
                Some(state) => {
                    if *fence != state.fence {
                        return Err(Error::Authority(format!(
                            "stale fence {fence} for {object_id} (current {})",
                            state.fence
                        )));
                    }
                    let expires = now_ns() + self.lease_ns;
                    Ok(vec![Message::LeaseGrant {
                        object_id: object_id.clone(),
                        epoch: self.epoch.value(),
                        fence: state.fence,
                        expires_ns: expires,
                    }])
                }
                None => Err(Error::NotFound(format!("object {object_id}"))),
            },
            Message::Migrate {
                object_id,
                new_owner,
                new_owner_addr,
                fence,
            } => {
                // The old owner requests migration with a fence at least the
                // object's current fence. The coordinator replies with a
                // Migrate instruction carrying the authoritative target address
                // and a bumped fence for the new owner.
                let state = self
                    .objects
                    .get(object_id)
                    .ok_or_else(|| Error::NotFound(format!("object {object_id}")))?;
                if *fence < state.fence {
                    return Err(Error::Authority(format!(
                        "stale fence {fence} for {object_id}"
                    )));
                }
                let to_fence = state.fence + 1;
                let target_addr = self
                    .nodes
                    .get(new_owner)
                    .cloned()
                    .unwrap_or_else(|| new_owner_addr.clone());
                self.migrations.insert(
                    object_id.clone(),
                    PendingMigration {
                        new_owner: new_owner.clone(),
                        to_fence,
                    },
                );
                Ok(vec![Message::Migrate {
                    object_id: object_id.clone(),
                    new_owner: new_owner.clone(),
                    new_owner_addr: target_addr,
                    fence: to_fence,
                }])
            }
            Message::MigrateAck {
                object_id,
                new_owner,
                fence,
            } => {
                let pending = self.migrations.get(object_id).ok_or_else(|| {
                    Error::Authority(format!("no pending migration for {object_id}"))
                })?;
                if pending.new_owner != *new_owner || pending.to_fence != *fence {
                    return Err(Error::Authority(format!(
                        "migration ack mismatch for {object_id}"
                    )));
                }
                if let Some(state) = self.objects.get_mut(object_id) {
                    state.owner = new_owner.clone();
                    state.fence = *fence;
                    state.epoch = self.epoch;
                }
                self.migrations.remove(object_id);
                Ok(vec![])
            }
            Message::Heartbeat { .. } => Ok(vec![]),
            _ => Ok(vec![Message::Error {
                code: "unsupported".into(),
                message: format!("coordinator does not handle {}", msg.msg_type().tag()),
            }]),
        }
    }
}

fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup_route() {
        let mut c = Coordinator::new(10_000_000);
        let regs = c
            .handle(&Message::Register {
                node_id: "n1".into(),
                addr: "127.0.0.1:1".into(),
            })
            .unwrap();
        assert_eq!(regs.len(), 1);
        if let Message::Hello { epoch, node_id, .. } = &regs[0] {
            assert_eq!(*epoch, 1);
            assert_eq!(node_id, "n1");
        } else {
            panic!("expected hello");
        }
        let creates = c
            .handle(&Message::Create {
                namespace: "ns".into(),
                key: "k".into(),
                generation: 1,
                byte_len: 8,
                compat: vec![],
                node_id: "n1".into(),
            })
            .unwrap();
        let oid = match &creates[0] {
            Message::CreateAck { object_id, .. } => object_id.clone(),
            _ => panic!("ack"),
        };
        let lk = c
            .handle(&Message::Lookup {
                namespace: "ns".into(),
                key: "k".into(),
                generation: 1,
                compat: vec![],
            })
            .unwrap();
        assert!(
            matches!(&lk[0], Message::LookupResult { found: true, owner: Some(o), owner_addr: Some(a), .. } if o == "n1" && a == "127.0.0.1:1")
        );
        let _ = oid;
    }

    #[test]
    fn epoch_advances_on_restart() {
        let mut c = Coordinator::new(10_000_000);
        let e0 = c.epoch();
        c.restart();
        assert!(c.epoch() > e0);
    }

    #[test]
    fn migrate_bumps_fence_and_transfers_ownership() {
        let mut c = Coordinator::new(10_000_000);
        c.handle(&Message::Register {
            node_id: "a".into(),
            addr: "127.0.0.1:1".into(),
        })
        .unwrap();
        c.handle(&Message::Register {
            node_id: "b".into(),
            addr: "127.0.0.1:2".into(),
        })
        .unwrap();
        let creates = c
            .handle(&Message::Create {
                namespace: "ns".into(),
                key: "k".into(),
                generation: 1,
                byte_len: 8,
                compat: vec![],
                node_id: "a".into(),
            })
            .unwrap();
        let oid = match &creates[0] {
            Message::CreateAck {
                object_id, fence, ..
            } => {
                assert_eq!(*fence, 0);
                object_id.clone()
            }
            _ => panic!("ack"),
        };
        // A requests migration to b with fence 0.
        let mig = c
            .handle(&Message::Migrate {
                object_id: oid.clone(),
                new_owner: "b".into(),
                new_owner_addr: "127.0.0.1:2".into(),
                fence: 0,
            })
            .unwrap();
        let to_fence = match &mig[0] {
            Message::Migrate { fence, .. } => *fence,
            _ => panic!("mig"),
        };
        assert!(to_fence > 0);
        let acks = c
            .handle(&Message::MigrateAck {
                object_id: oid.clone(),
                new_owner: "b".into(),
                fence: to_fence,
            })
            .unwrap();
        assert!(acks.is_empty());
        // Owner is now b; stale fence from a must be rejected.
        let renew = c.handle(&Message::LeaseRenew {
            object_id: oid,
            fence: 0,
        });
        assert!(renew.is_err());
    }
}
