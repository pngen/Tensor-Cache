//! Example 9: reconstruction economics.
//!
//! Demonstrates a case where transfer wins and a case where recomputation
//! wins, driven by the deterministic cost model.

mod common;
use tensorcache::cost::CostModel;
use tensorcache::planner::{decide, Action};
use tensorcache::tiers::Tier;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default model: device/host bandwidth is far higher than recompute, so
    // transferring a small object is cheaper than recomputing it.
    let m = CostModel::default();
    let plan = decide(&m, 1024, &Tier::Persistent, &[Tier::Host], true);
    println!(
        "small object -> chosen={:?} cost={} ns (transfer wins)",
        plan.action, plan.cost_ns
    );
    assert!(matches!(
        plan.action,
        Action::Transfer(_) | Action::ReuseInPlace
    ));

    // Tuned model: very slow persistent write + very cheap recompute, so for a
    // large object recomputation wins over a costly transfer.
    let slow = CostModel {
        storage_write_bw: 1_000_000,
        recompute_ns_per_byte: 1,
        recompute_base_ns: 1_000,
        ..Default::default()
    };
    let plan2 = decide(&slow, 1_000_000, &Tier::Persistent, &[Tier::Host], true);
    println!(
        "large object (slow store) -> chosen={:?} cost={} ns (recompute wins)",
        plan2.action, plan2.cost_ns
    );
    assert_eq!(plan2.action, Action::Reconstruct);

    // The plan is deterministic: same inputs give the same outcome.
    let a = decide(&slow, 1_000_000, &Tier::Persistent, &[Tier::Host], true);
    assert_eq!(a.action, plan2.action);
    println!("planner decision is deterministic");
    Ok(())
}
