//! Example 8: deduplication.
//!
//! Two logical tensor objects share byte-identical blocks. Physical storage is
//! shared (real savings) while the logical identities and compatibility remain
//! distinct.

mod common;
use tensorcache::runtime::TensorCache;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tc = TensorCache::new(common::config(1 << 24, None))?;
    let compat = common::f32(vec![16, 16], "dedup"); // 1024 bytes
    let data = common::payload(1024);

    let o1 = tc.register("ns", "obj-a", 1, compat.clone(), &data)?;
    let o2 = tc.register("ns", "obj-b", 1, compat.clone(), &data)?;
    let res = tc.resources();

    println!("object-a={}  object-b={}", o1, o2);
    println!(
        "logical objects={} host_used={} bytes",
        res.object_count, res.host_used
    );
    println!(
        "sum of logical sizes = {} bytes; physical shared = {} bytes",
        2 * 1024,
        res.host_used
    );
    assert_eq!(res.host_used, 1024, "dedup should share physical bytes");
    assert_eq!(res.object_count, 2, "identities must not collapse");

    // Deleting one object does not free the still-referenced shared block.
    tc.delete(&o1)?;
    let res2 = tc.resources();
    println!(
        "after deleting object-a: host_used={} objects={}",
        res2.host_used, res2.object_count
    );
    assert_eq!(res2.host_used, 1024);
    tc.delete(&o2)?;
    let res3 = tc.resources();
    assert_eq!(res3.host_used, 0);
    println!("after deleting both: host_used=0 (blocks reclaimed)");
    Ok(())
}
