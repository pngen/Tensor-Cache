#![forbid(unsafe_code)]
//! Storage tiers and tier classification.
//!
//! Tensor Cache distinguishes three physical tiers: accelerator, host memory
//! and persistent storage. The tier abstraction is deliberately small; the
//! residency machinery and the cost model work in terms of these tiers so that
//! movement, economics and accounting stay uniform.

use crate::backend::BackendId;

/// A physical storage tier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Tier {
    /// A device-resident buffer on a specific accelerator backend.
    Accelerator(BackendId),
    /// Host (system) memory, managed as a content-addressed block arena.
    Host,
    /// Durable persistent storage (local filesystem).
    Persistent,
}

/// The coarse tier kind, used by the cost model and accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierKind {
    Accelerator,
    Host,
    Persistent,
}

impl Tier {
    pub fn kind(&self) -> TierKind {
        match self {
            Tier::Accelerator(_) => TierKind::Accelerator,
            Tier::Host => TierKind::Host,
            Tier::Persistent => TierKind::Persistent,
        }
    }

    pub fn is_accelerator(&self) -> bool {
        matches!(self, Tier::Accelerator(_))
    }

    pub fn label(&self) -> String {
        match self {
            Tier::Accelerator(b) => format!("accelerator/{b}"),
            Tier::Host => "host".to_string(),
            Tier::Persistent => "persistent".to_string(),
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}
