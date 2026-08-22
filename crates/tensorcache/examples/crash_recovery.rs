//! Example 7: crash recovery.
//!
//! A tensor is persisted durably, then a fresh runtime (simulating a restart)
//! recovers it from disk and serves a valid object.

mod common;
use tensorcache::ident::Address;
use tensorcache::runtime::TensorCache;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = common::temp_dir("recovery");

    // "Process" 1: register + persist, then exit (simulated crash).
    {
        let tc = TensorCache::new(common::config(1 << 24, Some(dir.clone())))?;
        let compat = common::f32(vec![64, 64], "recovery");
        let payload = common::payload(64 * 64 * 4);
        let oid = tc.register("ns", "persisted", 1, compat.clone(), &payload)?;
        tc.persist(&oid)?;
        println!(
            "persisted {} bytes durably (process 1 exits)",
            payload.len()
        );
    }

    // "Process" 2: reopen the same store and recover.
    {
        let tc = TensorCache::new(common::config(1 << 24, Some(dir.clone())))?;
        let oid = Address::new("ns", "persisted", 1).object_id();
        let meta = tc.metadata(&oid)?;
        println!(
            "recovered object: durable={} placements={:?}",
            meta.durable, meta.placements
        );
        let bytes = tc.restore(&oid, &tensorcache::tiers::Tier::Host)?;
        assert_eq!(bytes, common::payload(64 * 64 * 4));
        tc.verify(&oid)?;
        println!("recovered and served valid object ({} bytes)", bytes.len());
    }

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
