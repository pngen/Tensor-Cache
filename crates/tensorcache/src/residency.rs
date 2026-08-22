#![forbid(unsafe_code)]
//! Residency state machine.
//!
//! Every tensor entry may be materialized in zero or more tiers. Its aggregate
//! residency is derived from the set of placements plus any transient movement
//! flags. The runtime never permits an illegal transition: e.g. it cannot
//! demote an object that is not host-resident, cannot evict the only copy of a
//! non-durable object that has no reconstruct path, and never leaves a tensor
//! in a phantom (absent-but-claimed) state.

use crate::tiers::Tier;

/// Aggregated residency of a tensor entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// Not materialized anywhere.
    Absent,
    /// Materialized only on an accelerator.
    Accelerator,
    /// Materialized only in host memory.
    Host,
    /// Materialized only in persistent storage.
    Persistent,
    /// Materialized in more than one tier.
    MultiResident,
}

/// Transient movement state held on an entry while a move is in flight.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoveFlags {
    pub migrating: bool,
    pub restoring: bool,
    pub evicting: bool,
    /// A quarantine flag: the entry failed integrity and must not be served.
    pub quarantined: bool,
    /// The entry was detected as invalid/corrupt and rejected.
    pub invalid: bool,
}

impl MoveFlags {
    pub fn healthy(&self) -> bool {
        !self.quarantined && !self.invalid
    }
}

/// Classify the aggregate residency from a set of placements.
pub fn classify(placements: &[Tier]) -> Residency {
    match placements.len() {
        0 => Residency::Absent,
        1 => match placements[0].kind() {
            crate::tiers::TierKind::Accelerator => Residency::Accelerator,
            crate::tiers::TierKind::Host => Residency::Host,
            crate::tiers::TierKind::Persistent => Residency::Persistent,
        },
        _ => Residency::MultiResident,
    }
}

/// Whether a tier currently holds a materialized copy.
pub fn is_resident(placements: &[Tier], tier: &Tier) -> bool {
    placements.iter().any(|p| p == tier)
}

/// Whether the entry has any placement at all.
pub fn any_present(placements: &[Tier]) -> bool {
    !placements.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendId;

    #[test]
    fn classify_basic() {
        assert_eq!(classify(&[]), Residency::Absent);
        assert_eq!(classify(&[Tier::Host]), Residency::Host);
        assert_eq!(classify(&[Tier::Persistent]), Residency::Persistent);
        assert_eq!(
            classify(&[Tier::Accelerator(BackendId::cpu(0))]),
            Residency::Accelerator
        );
        assert_eq!(
            classify(&[Tier::Host, Tier::Persistent]),
            Residency::MultiResident
        );
    }

    #[test]
    fn resident_helpers() {
        let placements = vec![Tier::Host, Tier::Persistent];
        assert!(is_resident(&placements, &Tier::Host));
        assert!(!is_resident(
            &placements,
            &Tier::Accelerator(BackendId::cpu(0))
        ));
        assert!(any_present(&placements));
        assert!(!any_present(&[]));
    }
}
