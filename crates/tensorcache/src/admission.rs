#![forbid(unsafe_code)]
//! Admission policy.
//!
//! Admission is bounded: it never permits unbounded memory growth and never
//! admits an object that cannot be satisfied by the current tier budget. The
//! policy is deterministic and inspectable; given the same candidate and the
//! same accounting state it always yields the same decision.

use crate::accounting::Accounting;
use crate::error::Error;
use crate::tiers::TierKind;

/// A candidate object under consideration for admission.
#[derive(Debug, Clone)]
pub struct AdmissionCandidate {
    pub object_id: String,
    pub bytes: u64,
    /// Cost to reconstruct the tensor (ns).
    pub reconstruction_cost_ns: u64,
    /// Cost to transfer it to the desired tier (ns).
    pub transfer_cost_ns: u64,
    /// Expected value of a future reuse (ns).
    pub reuse_value_ns: u64,
    pub priority: u32,
    pub desired_tier: TierKind,
    pub immutable: bool,
}

/// The admission policy.
#[derive(Debug, Clone)]
pub struct AdmissionPolicy {
    /// Reject any object larger than this many bytes.
    pub max_object_bytes: u64,
    /// Minimum expected reuse value to admit, regardless of size (ns).
    pub min_reuse_value_ns: u64,
    /// Required reuse value per admitted byte (ns).
    pub value_per_byte_ns: u64,
    /// Optional cap on the number of cached objects.
    pub max_objects: Option<u64>,
    /// Optional minimum reconstruction cost for an object to be worth caching.
    pub min_reconstruction_cost_ns: u64,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        AdmissionPolicy {
            max_object_bytes: 1 << 30, // 1 GiB
            min_reuse_value_ns: 250_000,
            value_per_byte_ns: 1,
            max_objects: None,
            min_reconstruction_cost_ns: 0,
        }
    }
}

/// The outcome of an admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Reject(String),
}

impl AdmissionDecision {
    pub fn is_admit(&self) -> bool {
        matches!(self, AdmissionDecision::Admit)
    }

    pub fn reason(&self) -> String {
        match self {
            AdmissionDecision::Admit => "admit".to_string(),
            AdmissionDecision::Reject(r) => r.clone(),
        }
    }
}

/// Evaluate the non-capacity admission criteria (size, value, object count).
pub fn evaluate_ignoring_capacity(
    policy: &AdmissionPolicy,
    candidate: &AdmissionCandidate,
    accounting: &Accounting,
) -> AdmissionDecision {
    if candidate.bytes > policy.max_object_bytes {
        return AdmissionDecision::Reject(format!(
            "object {} bytes exceeds max {}",
            candidate.bytes, policy.max_object_bytes
        ));
    }
    if candidate.reuse_value_ns < policy.min_reuse_value_ns {
        return AdmissionDecision::Reject(format!(
            "object reuse value {} below minimum {}",
            candidate.reuse_value_ns, policy.min_reuse_value_ns
        ));
    }
    // A reconstruction that is trivially cheap is not worth caching.
    if candidate.reconstruction_cost_ns < policy.min_reconstruction_cost_ns {
        return AdmissionDecision::Reject(format!(
            "object reconstruction cost {} below minimum {}",
            candidate.reconstruction_cost_ns, policy.min_reconstruction_cost_ns
        ));
    }
    // Value per byte requirement.
    let required_value = (candidate.bytes as u128)
        .saturating_mul(policy.value_per_byte_ns as u128)
        .min(u64::MAX as u128) as u64;
    if candidate.reuse_value_ns < required_value {
        return AdmissionDecision::Reject(format!(
            "object reuse value {} below per-byte requirement {}",
            candidate.reuse_value_ns, required_value
        ));
    }
    if let Some(max) = policy.max_objects {
        if accounting.object_count() >= max {
            return AdmissionDecision::Reject("object count limit reached".to_string());
        }
    }
    AdmissionDecision::Admit
}

/// Evaluate admission for a candidate against a policy and current accounting.
pub fn evaluate(
    policy: &AdmissionPolicy,
    candidate: &AdmissionCandidate,
    accounting: &Accounting,
) -> AdmissionDecision {
    let d = evaluate_ignoring_capacity(policy, candidate, accounting);
    if d.is_admit() {
        let free = accounting.free(candidate.desired_tier);
        if candidate.bytes > free {
            return AdmissionDecision::Reject(format!(
                "no capacity: need {} bytes, {} free in tier",
                candidate.bytes, free
            ));
        }
    }
    d
}

/// Error constructor helper used by the runtime when admission is rejected.
pub fn admission_error(decision: &AdmissionDecision) -> Error {
    Error::AdmissionRejected(decision.reason())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(bytes: u64, reuse: u64) -> AdmissionCandidate {
        AdmissionCandidate {
            object_id: "oid".into(),
            bytes,
            reconstruction_cost_ns: 1_000_000,
            transfer_cost_ns: 10_000,
            reuse_value_ns: reuse,
            priority: 0,
            desired_tier: TierKind::Host,
            immutable: true,
        }
    }

    fn accounting(cap: u64) -> Accounting {
        let mut a = Accounting::new();
        a.set_tier_capacity(TierKind::Host, cap);
        a
    }

    #[test]
    fn admission_when_capacity_and_value_suffice() {
        let p = AdmissionPolicy::default();
        let a = accounting(1 << 30);
        assert!(evaluate(&p, &candidate(1000, 1_000_000), &a).is_admit());
    }

    #[test]
    fn admission_rejects_too_large() {
        let p = AdmissionPolicy {
            max_object_bytes: 100,
            ..Default::default()
        };
        let a = accounting(1 << 30);
        let d = evaluate(&p, &candidate(1000, 1_000_000), &a);
        assert!(!d.is_admit());
    }

    #[test]
    fn admission_rejects_low_value() {
        let p = AdmissionPolicy {
            min_reuse_value_ns: 500_000,
            ..Default::default()
        };
        let a = accounting(1 << 30);
        let d = evaluate(&p, &candidate(1000, 10_000), &a);
        assert!(!d.is_admit());
    }

    #[test]
    fn admission_rejects_no_capacity() {
        let p = AdmissionPolicy::default();
        let a = accounting(100);
        let d = evaluate(&p, &candidate(1000, 1_000_000), &a);
        assert!(!d.is_admit());
    }

    #[test]
    fn admission_deterministic() {
        let p = AdmissionPolicy::default();
        let a = accounting(1 << 30);
        let c = candidate(1000, 1_000_000);
        assert_eq!(evaluate(&p, &c, &a), evaluate(&p, &c, &a));
    }
}
