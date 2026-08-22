//! Example 4: capacity pressure.
//!
//! A bounded cache is filled past capacity; admission triggers deterministic
//! eviction of the least valuable entries and the cache stays within budget.

mod common;
use tensorcache::admission::{evaluate_ignoring_capacity, AdmissionCandidate, AdmissionPolicy};
use tensorcache::runtime::TensorCache;
use tensorcache::tiers::TierKind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cs = common::startup_cost();
    let ns = "ns-pressure";
    let tc = TensorCache::new(common::config(8192, None))?;

    let compat = common::f32(vec![32, 32], "pressure"); // 4096 bytes each
    let mut admitted = 0;
    let mut evicted = 0;
    for i in 0..12 {
        // Distinct payloads so blocks are not deduplicated; this forces real
        // capacity pressure and deterministic eviction.
        let data: Vec<u8> = (0..4096).map(|j| ((i * 13 + j) % 251) as u8).collect();
        match tc.register(ns, format!("t{i}"), 1, compat.clone(), &data) {
            Ok(_) => admitted += 1,
            Err(tensorcache::error::Error::AdmissionRejected(_)) => evicted += 1,
            Err(_) => panic!("unexpected"),
        }
    }
    let res = tc.resources();
    println!("admitted={admitted} rejected_under_pressure={evicted}");
    println!(
        "host_used={} host_capacity={}",
        res.host_used, res.host_capacity
    );
    assert!(res.host_used <= res.host_capacity, "capacity overshoot!");
    println!("bounded cache stays within configured capacity");

    // Show the admission policy decision rationale for a candidate.
    let pol = AdmissionPolicy::default();
    let cand = AdmissionCandidate {
        object_id: "oid".into(),
        bytes: 4096,
        reconstruction_cost_ns: cs.reconstruct_cost_ns(4096),
        transfer_cost_ns: cs.transfer_cost_ns(
            &tensorcache::tiers::Tier::Host,
            &tensorcache::tiers::Tier::Host,
            4096,
        ),
        reuse_value_ns: cs.reconstruct_cost_ns(4096),
        priority: 0,
        desired_tier: TierKind::Host,
        immutable: true,
    };
    let mut acc = tensorcache::accounting::Accounting::new();
    acc.set_tier_capacity(TierKind::Host, 8192);
    let d = evaluate_ignoring_capacity(&pol, &cand, &acc);
    println!("deterministic admission decision: {}", d.is_admit());
    Ok(())
}
