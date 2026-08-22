#![forbid(unsafe_code)]
//! Exact resource accounting.
//!
//! Accounting is authoritative and must return to its expected value after
//! eviction, reclamation, migration, failed transfer, delete, shutdown, failed
//! admission and every allocation/free cycle. Every mutation is checked so that
//! bytes never go negative and capacity is never overshot. Reservations are
//! tracked separately from committed usage so a pending transfer cannot push a
//! tier over its budget.

use crate::error::{Error, Result};
use crate::tiers::TierKind;

/// Per-tier byte budget.
#[derive(Debug, Default, Clone)]
pub struct Budget {
    capacity: u64,
    used: u64,
    reserved: u64,
}

impl Budget {
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    /// Free (unused and unreserved) bytes.
    pub fn free(&self) -> u64 {
        self.capacity
            .saturating_sub(self.used.saturating_add(self.reserved))
    }

    fn set_capacity(&mut self, cap: u64) {
        self.capacity = cap;
    }

    fn add_used(&mut self, n: u64) -> Result<()> {
        let next = self
            .used
            .checked_add(n)
            .ok_or_else(|| Error::Accounting("tier byte usage overflow".into()))?;
        if next > self.capacity {
            return Err(Error::Accounting(format!(
                "tier capacity overshoot: used {} + {} > {}",
                self.used, n, self.capacity
            )));
        }
        self.used = next;
        Ok(())
    }

    fn sub_used(&mut self, n: u64) -> Result<()> {
        if n > self.used {
            return Err(Error::Accounting(format!(
                "tier byte underflow: remove {n} > used {}",
                self.used
            )));
        }
        self.used -= n;
        Ok(())
    }

    fn reserve(&mut self, n: u64) -> Result<()> {
        let next_reserved = self
            .reserved
            .checked_add(n)
            .ok_or_else(|| Error::Accounting("tier reservation overflow".into()))?;
        // Capacity check accounts for both used and reserved.
        if self.used.saturating_add(next_reserved) > self.capacity {
            return Err(Error::Accounting(format!(
                "tier reservation would overshoot: used {} + reserved {} + {} > {}",
                self.used, self.reserved, n, self.capacity
            )));
        }
        self.reserved = next_reserved;
        Ok(())
    }

    fn commit_reserve(&mut self, n: u64) -> Result<()> {
        if n > self.reserved {
            return Err(Error::Accounting("commit exceeds reservation".into()));
        }
        self.reserved -= n;
        self.add_used(n)
    }

    fn cancel_reserve(&mut self, n: u64) -> Result<()> {
        if n > self.reserved {
            return Err(Error::Accounting("cancel exceeds reservation".into()));
        }
        self.reserved -= n;
        Ok(())
    }
}

/// Global, exact resource accounting for a runtime.
#[derive(Debug, Default, Clone)]
pub struct Accounting {
    accel: Budget,
    host: Budget,
    storage: Budget,
    objects: u64,
    blocks: u64,
    replicas: u64,
}

impl Accounting {
    pub fn new() -> Self {
        Accounting::default()
    }

    pub fn set_tier_capacity(&mut self, kind: TierKind, cap: u64) {
        match kind {
            TierKind::Accelerator => self.accel.set_capacity(cap),
            TierKind::Host => self.host.set_capacity(cap),
            TierKind::Persistent => self.storage.set_capacity(cap),
        }
    }

    pub fn capacity(&self, kind: TierKind) -> u64 {
        match kind {
            TierKind::Accelerator => self.accel.capacity(),
            TierKind::Host => self.host.capacity(),
            TierKind::Persistent => self.storage.capacity(),
        }
    }

    pub fn used(&self, kind: TierKind) -> u64 {
        match kind {
            TierKind::Accelerator => self.accel.used(),
            TierKind::Host => self.host.used(),
            TierKind::Persistent => self.storage.used(),
        }
    }

    pub fn reserved(&self, kind: TierKind) -> u64 {
        match kind {
            TierKind::Accelerator => self.accel.reserved(),
            TierKind::Host => self.host.reserved(),
            TierKind::Persistent => self.storage.reserved(),
        }
    }

    pub fn free(&self, kind: TierKind) -> u64 {
        match kind {
            TierKind::Accelerator => self.accel.free(),
            TierKind::Host => self.host.free(),
            TierKind::Persistent => self.storage.free(),
        }
    }

    pub fn add_bytes(&mut self, kind: TierKind, n: u64) -> Result<()> {
        match kind {
            TierKind::Accelerator => self.accel.add_used(n),
            TierKind::Host => self.host.add_used(n),
            TierKind::Persistent => self.storage.add_used(n),
        }
    }

    pub fn sub_bytes(&mut self, kind: TierKind, n: u64) -> Result<()> {
        match kind {
            TierKind::Accelerator => self.accel.sub_used(n),
            TierKind::Host => self.host.sub_used(n),
            TierKind::Persistent => self.storage.sub_used(n),
        }
    }

    pub fn reserve(&mut self, kind: TierKind, n: u64) -> Result<()> {
        match kind {
            TierKind::Accelerator => self.accel.reserve(n),
            TierKind::Host => self.host.reserve(n),
            TierKind::Persistent => self.storage.reserve(n),
        }
    }

    pub fn commit_reserve(&mut self, kind: TierKind, n: u64) -> Result<()> {
        match kind {
            TierKind::Accelerator => self.accel.commit_reserve(n),
            TierKind::Host => self.host.commit_reserve(n),
            TierKind::Persistent => self.storage.commit_reserve(n),
        }
    }

    pub fn cancel_reserve(&mut self, kind: TierKind, n: u64) -> Result<()> {
        match kind {
            TierKind::Accelerator => self.accel.cancel_reserve(n),
            TierKind::Host => self.host.cancel_reserve(n),
            TierKind::Persistent => self.storage.cancel_reserve(n),
        }
    }

    pub fn object_count(&self) -> u64 {
        self.objects
    }
    pub fn set_object_count(&mut self, n: u64) {
        self.objects = n;
    }
    pub fn block_count(&self) -> u64 {
        self.blocks
    }
    pub fn set_block_count(&mut self, n: u64) {
        self.blocks = n;
    }
    pub fn replica_count(&self) -> u64 {
        self.replicas
    }
    pub fn set_replica_count(&mut self, n: u64) {
        self.replicas = n;
    }

    /// Total bytes across all tiers (for reporting).
    pub fn total_bytes(&self) -> u64 {
        self.accel
            .used()
            .saturating_add(self.host.used())
            .saturating_add(self.storage.used())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounting_add_sub_is_exact() {
        let mut a = Accounting::new();
        a.set_tier_capacity(TierKind::Host, 1000);
        a.add_bytes(TierKind::Host, 300).unwrap();
        a.add_bytes(TierKind::Host, 200).unwrap();
        assert_eq!(a.used(TierKind::Host), 500);
        a.sub_bytes(TierKind::Host, 500).unwrap();
        assert_eq!(a.used(TierKind::Host), 0);
    }

    #[test]
    fn accounting_never_negative() {
        let mut a = Accounting::new();
        a.set_tier_capacity(TierKind::Host, 1000);
        a.add_bytes(TierKind::Host, 100).unwrap();
        assert!(a.sub_bytes(TierKind::Host, 101).is_err());
    }

    #[test]
    fn accounting_never_overshoot() {
        let mut a = Accounting::new();
        a.set_tier_capacity(TierKind::Host, 100);
        assert!(a.add_bytes(TierKind::Host, 101).is_err());
    }

    #[test]
    fn reservation_flow() {
        let mut a = Accounting::new();
        a.set_tier_capacity(TierKind::Host, 100);
        a.reserve(TierKind::Host, 60).unwrap();
        assert_eq!(a.free(TierKind::Host), 40);
        a.commit_reserve(TierKind::Host, 60).unwrap();
        assert_eq!(a.used(TierKind::Host), 60);
        assert_eq!(a.reserved(TierKind::Host), 0);
        // Cannot reserve beyond the remaining 40.
        assert!(a.reserve(TierKind::Host, 41).is_err());
    }

    #[test]
    fn cancel_reserve_restores_free() {
        let mut a = Accounting::new();
        a.set_tier_capacity(TierKind::Host, 100);
        a.reserve(TierKind::Host, 30).unwrap();
        a.cancel_reserve(TierKind::Host, 30).unwrap();
        assert_eq!(a.free(TierKind::Host), 100);
    }
}
