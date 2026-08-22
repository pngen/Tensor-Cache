//! Adversarial hardening probes: accounting invariants, dedup reference
//! correctness, and concurrent access safety.

use std::path::PathBuf;
use std::sync::Arc;

use tensorcache::backend::BackendId;
use tensorcache::compat::CompatKey;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::runtime::{RuntimeConfig, TensorCache};
use tensorcache::tiers::Tier;

fn compat() -> CompatKey {
    CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![16, 16]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("hardening".into()),
        ..Default::default()
    }
}

fn payload(seed: u8, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(31).wrapping_add(seed as usize)) as u8)
        .collect()
}

fn temp_dir(tag: &str) -> PathBuf {
    static C: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = C.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::env::temp_dir().join(format!("tc-hard-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// After a full lifecycle the per-tier byte accounting must return to zero and
/// never go negative.
#[test]
fn accounting_returns_to_zero_after_lifecycle() {
    let dir = temp_dir("life");
    let cfg = RuntimeConfig {
        host_capacity: 1 << 20,
        persistent_path: Some(dir.clone()),
        ..Default::default()
    };
    let tc = TensorCache::new(cfg).unwrap();
    let accel = Tier::Accelerator(BackendId::cpu(0));
    let c = compat();
    let data = payload(7, 1024);

    let oid = tc.register("ns", "a", 0, c.clone(), &data).unwrap();
    tc.persist(&oid).unwrap();
    tc.promote(&oid, &accel).unwrap();
    let r = tc.resources();
    assert!(r.host_used >= 1024);
    assert!(r.accel_used >= 1024);
    assert!(r.storage_used >= 1024);

    tc.delete(&oid).unwrap();
    let r = tc.resources();
    assert_eq!(r.host_used, 0);
    assert_eq!(r.accel_used, 0);
    assert_eq!(r.storage_used, 0);
    assert_eq!(r.object_count, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A failed admission (capacity) must not leave accounting in a bad state.
#[test]
fn failed_admission_leaves_accounting_exact() {
    let cfg = RuntimeConfig {
        host_capacity: 4096,
        ..Default::default()
    };
    let tc = TensorCache::new(cfg).unwrap();
    let c = compat();
    for i in 0..10 {
        let data = payload(i as u8, 4096);
        let _ = tc.register("ns", format!("t{i}"), 0, c.clone(), &data);
    }
    // Never overshoot capacity.
    let r = tc.resources();
    assert!(r.host_used <= 4096, "overshoot: {}", r.host_used);
    assert!(r.host_reserved == 0, "no leaked reservations");
}

/// Deduplication must not leak references: after deleting all referring
/// objects, the block bytes are reclaimed exactly once.
#[test]
fn dedup_reference_leak_after_delete() {
    let cfg = RuntimeConfig {
        host_capacity: 1 << 20,
        ..Default::default()
    };
    let tc = TensorCache::new(cfg).unwrap();
    let c = compat();
    let data = payload(3, 1024);
    let mut oids = Vec::new();
    for i in 0..4 {
        oids.push(
            tc.register("ns", format!("dup{i}"), 0, c.clone(), &data)
                .unwrap(),
        );
    }
    // All share one physical block.
    assert_eq!(tc.resources().host_used, 1024);
    // Deleting one at a time frees nothing until the last reference is gone.
    tc.delete(&oids[0]).unwrap();
    assert_eq!(tc.resources().host_used, 1024);
    tc.delete(&oids[1]).unwrap();
    tc.delete(&oids[2]).unwrap();
    assert_eq!(tc.resources().host_used, 1024);
    tc.delete(&oids[3]).unwrap();
    assert_eq!(tc.resources().host_used, 0);
    assert_eq!(tc.resources().block_count, 0);
}

/// Concurrent readers + writers on a single runtime must not deadlock or corrupt
/// accounting.
#[test]
fn concurrent_access_is_safe() {
    let cfg = RuntimeConfig {
        host_capacity: 1 << 22,
        ..Default::default()
    };
    let tc = Arc::new(TensorCache::new(cfg).unwrap());
    let mut handles = Vec::new();
    for t in 0..8u8 {
        let tc = Arc::clone(&tc);
        let c = compat();
        handles.push(std::thread::spawn(move || {
            for i in 0..50u64 {
                let data = payload(t, 1024);
                let key = format!("t{t}-{i}");
                let _ = tc.register("ns", &key, 0, c.clone(), &data);
                let _ = tc.lookup("ns", &key, 0, &c);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // No panic / deadlock; accounting is internally consistent.
    let r = tc.resources();
    assert!(r.host_used <= r.host_capacity);
    assert!(r.object_count > 0);
    assert_eq!(
        r.host_reserved, 0,
        "no leaked reservations after concurrent work"
    );
}
