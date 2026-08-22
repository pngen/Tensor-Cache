#![forbid(unsafe_code)]
//! Deterministic reuse/placement planner.
//!
//! Given a destination tier, the set of tiers where a tensor is currently
//! resident, its byte length and whether it can be reconstructed, the planner
//! chooses the lowest-cost action among: reuse in place, transfer from another
//! tier, restore from persistent storage, reconstruct/recompute, or reject.
//!
//! The decision is fully deterministic: costs are integer nanoseconds from the
//! cost model, candidate actions are compared by ascending cost, and ties are
//! broken by a stable action priority. No heuristic or fuzzy compatibility is
//! involved.

use crate::cost::CostModel;
use crate::tiers::Tier;

/// The action chosen by the planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    ReuseInPlace,
    Transfer(Tier),
    Restore,
    Reconstruct,
    Reject,
}

/// A planned decision with its estimated cost and a human-readable rationale.
#[derive(Debug, Clone)]
pub struct Plan {
    pub action: Action,
    pub cost_ns: u64,
    pub rationale: String,
}

fn action_priority(a: &Action) -> u8 {
    match a {
        Action::ReuseInPlace => 0,
        Action::Transfer(_) => 1,
        Action::Restore => 2,
        Action::Reconstruct => 3,
        Action::Reject => 4,
    }
}

/// Decide the best placement/reuse action.
///
/// - `dest`: the tier a reuse would materialize into.
/// - `placements`: the tiers that currently hold the tensor.
/// - `can_reconstruct`: whether the owner can recompute the tensor.
pub fn decide(
    cost: &CostModel,
    bytes: u64,
    dest: &Tier,
    placements: &[Tier],
    can_reconstruct: bool,
) -> Plan {
    // Reuse in place if the destination already holds a copy.
    if placements.iter().any(|p| p == dest) {
        return Plan {
            action: Action::ReuseInPlace,
            cost_ns: cost.reuse_in_place_cost_ns(),
            rationale: format!("already {} resident", dest.label()),
        };
    }

    let mut candidates: Vec<(u64, Action)> = Vec::new();

    // Transfer from each other tier that holds a copy.
    for p in placements {
        if p != dest {
            let c = cost.transfer_cost_ns(p, dest, bytes);
            candidates.push((c, Action::Transfer(p.clone())));
        }
    }

    // Restore from persistent storage is a dedicated high-latency path.
    if placements
        .iter()
        .any(|p| p.kind() == crate::tiers::TierKind::Persistent)
    {
        let c = cost.restore_cost_ns(dest, bytes);
        candidates.push((c, Action::Restore));
    }

    // Reconstruction is always an option if the owner can recompute.
    if can_reconstruct {
        let c = cost.reconstruct_cost_ns(bytes);
        candidates.push((c, Action::Reconstruct));
    }

    if candidates.is_empty() {
        return Plan {
            action: Action::Reject,
            cost_ns: u64::MAX,
            rationale: "no source placement and reconstruction not possible".to_string(),
        };
    }

    // Deterministic minimum: lowest cost, tie-break by action priority.
    candidates.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| action_priority(&a.1).cmp(&action_priority(&b.1)))
    });
    let (cost_ns, action) = candidates.remove(0);
    let rationale = match &action {
        Action::ReuseInPlace => "reuse in place".to_string(),
        Action::Transfer(t) => format!("transfer from {}", t.label()),
        Action::Restore => "restore from persistent storage".to_string(),
        Action::Reconstruct => "reconstruct/recompute".to_string(),
        Action::Reject => "reject".to_string(),
    };
    Plan {
        action,
        cost_ns,
        rationale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiers::Tier;

    #[test]
    fn reuse_in_place_always_wins_when_resident() {
        let m = CostModel::default();
        let p = decide(&m, 1_000_000, &Tier::Host, &[Tier::Host], false);
        assert_eq!(p.action, Action::ReuseInPlace);
        assert_eq!(p.cost_ns, 0);
    }

    #[test]
    fn transfer_chosen_when_host_present() {
        let m = CostModel::default();
        let p = decide(&m, 10_000, &Tier::Persistent, &[Tier::Host], false);
        assert!(matches!(p.action, Action::Transfer(_)));
        assert_eq!(
            p.cost_ns,
            m.transfer_cost_ns(&Tier::Host, &Tier::Persistent, 10_000)
        );
    }

    #[test]
    fn reconstruct_wins_when_transfer_is_slow() {
        let m = CostModel {
            storage_write_bw: 1_000_000,
            recompute_ns_per_byte: 1,
            recompute_base_ns: 1_000,
            ..Default::default()
        };
        let bytes = 1_000_000;
        let p = decide(&m, bytes, &Tier::Persistent, &[Tier::Host], true);
        assert_eq!(p.action, Action::Reconstruct);
    }

    #[test]
    fn reject_when_no_source_and_cannot_reconstruct() {
        let m = CostModel::default();
        let p = decide(&m, 10, &Tier::Persistent, &[], false);
        assert_eq!(p.action, Action::Reject);
    }

    #[test]
    fn deterministic_with_multiple_sources() {
        let m = CostModel::default();
        let p1 = decide(
            &m,
            10_000,
            &Tier::Persistent,
            &[
                Tier::Host,
                Tier::Accelerator(crate::backend::BackendId::cpu(0)),
            ],
            false,
        );
        let p2 = decide(
            &m,
            10_000,
            &Tier::Persistent,
            &[
                Tier::Host,
                Tier::Accelerator(crate::backend::BackendId::cpu(0)),
            ],
            false,
        );
        assert_eq!(p1.action, p2.action);
        assert_eq!(p1.cost_ns, p2.cost_ns);
    }
}
