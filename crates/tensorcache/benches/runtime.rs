//! Tensor Cache runtime benchmark (run in release mode).
//!
//! Measures the meaningful runtime operations on the host tier. This is a
//! custom harness (no external crate) that reports per-op latency and
//! throughput for the default CPU/host configuration. See BENCHMARKS.md for the
//! environment and how to interpret the numbers.

use std::path::PathBuf;
use std::time::Instant;

use tensorcache::admission::{evaluate_ignoring_capacity, AdmissionCandidate, AdmissionPolicy};
use tensorcache::backend::BackendId;
use tensorcache::compat::CompatKey;
use tensorcache::crc::crc32c;
use tensorcache::dtype::Dtype;
use tensorcache::eviction::{eviction_order, Evictable, EvictionPolicy};
use tensorcache::geometry::{Layout, Shape};
use tensorcache::hash::hash;
use tensorcache::planner::decide;
use tensorcache::runtime::{RuntimeConfig, TensorCache};
use tensorcache::tiers::{Tier, TierKind};

fn report(name: &str, bytes: usize, iters: usize, elapsed: std::time::Duration) {
    let per_ns = elapsed.as_nanos() as f64 / iters as f64;
    let through = bytes as f64 / (elapsed.as_secs_f64().max(1e-9));
    println!("{name:<34} iters={:<7} bytes={:<9} total={:>11?}  per-op={:>10.1}ns  throughput={:>12.1} B/s", iters, bytes, elapsed, per_ns, through);
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tc-bench-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn main() {
    let bytes: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    println!("Tensor Cache benchmark (release)");
    println!("  op_size={bytes} bytes  iterations={iters}");
    println!("  hardware=CPU host (no accelerator transfers below)");
    println!();

    let compat = CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![bytes as u64 / 4]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("bench".into()),
        ..Default::default()
    };
    let payload: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    let dir = temp_dir("ste");
    let cfg = RuntimeConfig {
        host_capacity: 1 << 30,
        persistent_path: Some(dir.clone()),
        ..Default::default()
    };
    let tc = TensorCache::new(cfg).unwrap();
    let accel = Tier::Accelerator(BackendId::cpu(0));
    let host = Tier::Host;

    // Dedup hashing throughput.
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = hash(&payload);
        let _ = crc32c(&payload);
    }
    report("dedup_hash_sha256+crc32c", bytes, iters, t0.elapsed());

    // Register: fresh object each iteration (bytes differ to avoid dedup).
    let t0 = Instant::now();
    for i in 0..iters {
        let data: Vec<u8> = (0..bytes).map(|j| ((i * 13 + j) % 251) as u8).collect();
        let _ = tc.register("bench", format!("r{i}"), 0, compat.clone(), &data);
    }
    report("create_register", bytes, iters, t0.elapsed());

    // Exact lookup hit (re-use the first registered address's key).
    let hit_key = tc
        .metadata(
            &tc.register("bench", "hit", 0, compat.clone(), &payload)
                .unwrap(),
        )
        .unwrap()
        .key;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = tc.lookup("bench", &hit_key, 0, &compat);
    }
    report("exact_lookup_hit", bytes, iters, t0.elapsed());

    // Cache miss.
    let t0 = Instant::now();
    for i in 0..iters {
        let _ = tc.lookup("bench", format!("missing-{i}"), 0, &compat);
    }
    report("cache_miss", bytes, iters, t0.elapsed());

    // Compat check (compat_id hashing).
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = compat.compat_id();
    }
    println!(
        "{:<34} iters={:<7} bytes={:<9} total={:>11?}  per-op={:>10.1}ns",
        "compat_check(compat_id)",
        iters,
        bytes,
        t0.elapsed(),
        t0.elapsed().as_nanos() as f64 / iters as f64
    );

    // Placement ops: operate on each fresh object exactly once.
    {
        let objs: Vec<_> = (0..iters)
            .map(|i| {
                let data: Vec<u8> = (0..bytes).map(|j| ((i * 29 + j) % 251) as u8).collect();
                tc.register("bench", format!("p{i}"), 0, compat.clone(), &data)
                    .unwrap()
            })
            .collect();
        let t0 = Instant::now();
        for o in &objs {
            let _ = tc.promote(o, &accel);
        }
        report("host_promote_to_accel", bytes, iters, t0.elapsed());

        let t0 = Instant::now();
        for o in &objs {
            let _ = tc.demote(o, &accel);
        }
        report("accel_demote_to_host", bytes, iters, t0.elapsed());

        let t0 = Instant::now();
        for o in &objs {
            let _ = tc.persist(o);
        }
        report("persist_to_storage", bytes, iters, t0.elapsed());

        let t0 = Instant::now();
        for o in &objs {
            let _ = tc.restore(o, &host);
        }
        report("restore_from_storage", bytes, iters, t0.elapsed());

        let t0 = Instant::now();
        for o in &objs {
            let _ = tc.verify(o);
        }
        report("integrity_verify", bytes, iters, t0.elapsed());
    }

    // Admission decision.
    let pol = AdmissionPolicy::default();
    let mut acc = tensorcache::accounting::Accounting::new();
    acc.set_tier_capacity(TierKind::Host, 1 << 30);
    let cand = AdmissionCandidate {
        object_id: "oid".into(),
        bytes: bytes as u64,
        reconstruction_cost_ns: 5_000_000,
        transfer_cost_ns: 2_000,
        reuse_value_ns: 6_000_000,
        priority: 0,
        desired_tier: TierKind::Host,
        immutable: true,
    };
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = evaluate_ignoring_capacity(&pol, &cand, &acc);
    }
    println!(
        "{:<34} iters={:<7} bytes={:<9} total={:>11?}  per-op={:>10.1}ns",
        "admission_decision",
        iters,
        bytes,
        t0.elapsed(),
        t0.elapsed().as_nanos() as f64 / iters as f64
    );

    // Eviction order.
    let pol2 = EvictionPolicy::default();
    let entries: Vec<Evictable> = (0..iters as u64)
        .map(|i| Evictable {
            object_id: i.to_string(),
            bytes: bytes as u64,
            reuse_count: i % 7,
            age_seconds: i % 100,
            reconstruction_cost_ns: 1_000_000,
            priority: 0,
            pressure: 0.5,
            durable: false,
        })
        .collect();
    let t0 = Instant::now();
    let _ = eviction_order(&pol2, &entries);
    println!(
        "{:<34} iters={:<7} bytes={:<9} total={:>11?}  per-op={:>10.1}ns",
        "eviction_order",
        entries.len(),
        bytes,
        t0.elapsed(),
        t0.elapsed().as_nanos() as f64 / entries.len() as f64
    );

    // Planner decision.
    let cost = tensorcache::cost::CostModel::default();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = decide(&cost, bytes as u64, &Tier::Persistent, &[Tier::Host], true);
    }
    println!(
        "{:<34} iters={:<7} bytes={:<9} total={:>11?}  per-op={:>10.1}ns",
        "planner_decision",
        iters,
        bytes,
        t0.elapsed(),
        t0.elapsed().as_nanos() as f64 / iters as f64
    );

    let _ = std::fs::remove_dir_all(&dir);
    println!("\nbenchmark complete (see BENCHMARKS.md for environment)");
}
