//! Example 3: tiered residency.
//!
//! A tensor moves accelerator -> host -> persistent storage, then is restored
//! and its integrity verified on the way back.

mod common;
use tensorcache::backend::BackendId;
use tensorcache::compat::CompatKey;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::runtime::TensorCache;
use tensorcache::tiers::Tier;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let persist_dir = common::temp_dir("tiered");
    let cfg = common::config(1 << 24, Some(persist_dir.clone()));
    let tc = TensorCache::new(cfg)?;
    let compat = CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![64 * 64]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("tier-demo".into()),
        ..Default::default()
    };
    let payload = common::payload(64 * 64 * 4); // 16384 bytes
    let oid = tc.register("ns", "tiered", 1, compat.clone(), &payload)?;

    let accel = Tier::Accelerator(BackendId::cpu(0));
    tc.promote(&oid, &accel)?;
    println!(
        "promoted to accelerator: {:?}",
        tc.metadata(&oid)?.placements
    );

    tc.promote(&oid, &Tier::Persistent)?;
    println!("persisted to disk: {:?}", tc.metadata(&oid)?.placements);
    println!("storage_used={}", tc.resources().storage_used);

    tc.evict(&oid)?;
    println!("after evict: {:?}", tc.metadata(&oid)?.placements);

    let bytes = tc.restore(&oid, &Tier::Host)?;
    assert_eq!(bytes, payload);
    tc.verify(&oid)?;
    println!("restored {} bytes and verified integrity", bytes.len());
    let _ = std::fs::remove_dir_all(&persist_dir);
    Ok(())
}
