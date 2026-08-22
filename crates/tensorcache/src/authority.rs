#![forbid(unsafe_code)]
//! Distributed authority: epochs, boot identity, ownership leases and fence
//! tokens.
//!
//! Authority is strictly ordered. A coordinator holds an epoch that advances on
//! every restart. Each node records the coordinator boot identity (a random id
//! generated per coordinator process) and the epoch under which it registered.
//! A mutation is legal only if the requester holds a *current* lease (not
//! expired) whose epoch and boot identity match the coordinator's, and whose
//! fence is at least as large as the current fence for the object. A stale
//! epoch, a stale boot identity, an expired lease or a low fence is rejected,
//! so migration cannot create dual authoritative owners and a restarted node
//! never silently inherits old authority.

/// A strictly monotonic coordinator epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Epoch(pub u64);

impl Epoch {
    pub fn new(v: u64) -> Self {
        Epoch(v)
    }
    pub fn next(self) -> Epoch {
        Epoch(self.0.saturating_add(1))
    }
    pub fn value(self) -> u64 {
        self.0
    }
}

/// The immutable boot identity of a coordinator process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootId(String);

impl BootId {
    pub fn new() -> Self {
        // A fresh random-like id for this process; sufficiently unique for
        // authority fencing without external randomness infrastructure.
        BootId(random_id())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for BootId {
    fn default() -> Self {
        BootId::new()
    }
}

impl std::fmt::Display for BootId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A node identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(String);

impl NodeId {
    pub fn new(s: impl Into<String>) -> Self {
        NodeId(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A mutation lease granting a node read-write authority over an object until a
/// deadline with a monotonically increasing fence token.
#[derive(Debug, Clone)]
pub struct Lease {
    pub owner: NodeId,
    pub epoch: Epoch,
    pub boot_id: BootId,
    pub fence: u64,
    pub expires_ns: u64,
}

impl Lease {
    /// Whether the lease is still valid against the coordinator's authority
    /// state. Rejects a stale epoch, a stale boot identity, an expired lease and
    /// a fence lower than the object's current fence.
    pub fn permits_mutation(
        &self,
        current_epoch: Epoch,
        current_boot: &BootId,
        object_fence: u64,
        now_ns: u64,
    ) -> bool {
        self.epoch == current_epoch
            && self.boot_id.as_str() == current_boot.as_str()
            && self.expires_ns > now_ns
            && self.fence >= object_fence
    }

    /// Whether the lease has been superseded (stale epoch/boot).
    pub fn is_stale_authority(&self, current_epoch: Epoch, current_boot: &BootId) -> bool {
        self.epoch != current_epoch || self.boot_id.as_str() != current_boot.as_str()
    }
}

/// A random-ish id derived from the monotonic clock plus a counter.
fn random_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    seed ^= c.rotate_left(17);
    // xorshift to spread bits.
    let mut x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    crate::hash::hex(&x.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_permits_when_current_and_unexpired_and_fenced() {
        let epoch = Epoch::new(5);
        let boot = BootId::new();
        let now = 1_000_000;
        let lease = Lease {
            owner: NodeId::new("a"),
            epoch,
            boot_id: boot.clone(),
            fence: 3,
            expires_ns: now + 100,
        };
        assert!(lease.permits_mutation(epoch, &boot, 3, now));
        assert!(lease.permits_mutation(epoch, &boot, 2, now));
    }

    #[test]
    fn stale_epoch_or_boot_rejected() {
        let boot = BootId::new();
        let now = 1_000_000;
        let lease = Lease {
            owner: NodeId::new("a"),
            epoch: Epoch::new(1),
            boot_id: boot.clone(),
            fence: 3,
            expires_ns: now + 100,
        };
        assert!(!lease.permits_mutation(Epoch::new(2), &boot, 3, now));
        assert!(!lease.permits_mutation(Epoch::new(1), &BootId::new(), 3, now));
    }

    #[test]
    fn expired_lease_and_stale_fence_rejected() {
        let epoch = Epoch::new(1);
        let boot = BootId::new();
        let now = 1_000_000;
        let expired = Lease {
            owner: NodeId::new("a"),
            epoch,
            boot_id: boot.clone(),
            fence: 3,
            expires_ns: now - 1,
        };
        assert!(!expired.permits_mutation(epoch, &boot, 3, now));
        let low_fence = Lease {
            owner: NodeId::new("a"),
            epoch,
            boot_id: boot.clone(),
            fence: 1,
            expires_ns: now + 100,
        };
        assert!(!low_fence.permits_mutation(epoch, &boot, 3, now));
    }

    #[test]
    fn epoch_is_monotonic() {
        let e = Epoch::new(4);
        assert_eq!(e.next().value(), 5);
        assert!(e < e.next());
    }

    #[test]
    fn boot_id_distinct_across_instances() {
        assert_ne!(BootId::new().as_str(), BootId::new().as_str());
    }
}
