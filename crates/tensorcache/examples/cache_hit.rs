//! Example 1: reusable embedding / activation cache hit.
//!
//! A produced tensor is registered; a later request reuses it instead of
//! recomputing, demonstrating the avoided reconstruction cost.

mod common;
use tensorcache::runtime::TensorCache;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = common::config(1 << 24, None);
    let tc = TensorCache::new(cfg)?;
    let compat = common::f32(vec![128, 4096], "embedder-v1");

    // Producer materializes an embedding tensor (expensive to reconstruct).
    let payload = common::payload(128 * 4096 * 4);
    let oid = tc.register("demo", "embeddings.layer3", 1, compat.clone(), &payload)?;
    println!("registered embedding {} ({} bytes)", oid, payload.len());

    // A later request reuses the same state.
    let hit = tc.lookup("demo", "embeddings.layer3", 1, &compat)?;
    println!(
        "lookup hit: source_tier={:?} bytes={}",
        hit.source_tier, hit.bytes
    );
    println!(
        "avoided reconstruction: {} ns",
        hit.reconstruction_avoided_ns
    );
    println!("transfer cost (if moved): {} ns", hit.transfer_cost_ns);

    // Verify the reused payload is bit-identical.
    let got = tc.restore(&oid, &tensorcache::tiers::Tier::Host)?;
    assert_eq!(got, payload);
    println!("reused payload verified bit-identical");
    Ok(())
}
