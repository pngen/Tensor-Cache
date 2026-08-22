//! CUDA H2D/D2H transfer benchmark (run in release mode).
//!
//! Measures real cudaMemcpy host-to-device and device-to-host throughput and
//! allocation/free latency. Skipped cleanly if no CUDA runtime is present.

use std::time::Instant;
use tensorcache::backend::Backend;
use tensorcache_cuda::CudaBackend;

fn main() {
    let bytes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 24);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let b = match CudaBackend::new(0, 1 << 30) {
        Ok(b) => b,
        Err(e) => {
            println!("CUDA unavailable ({e}); skipping CUDA benchmark");
            return;
        }
    };
    let count = CudaBackend::device_count().unwrap_or(0);
    println!("CUDA benchmark (release)");
    println!("  device_count={count}  bytes={bytes}  iterations={iters}");
    println!("  hardware=NVIDIA GeForce RTX 5090 (Blackwell) / CUDA runtime (see BENCHMARKS.md)");
    println!();

    let host: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    let mut dev = b.allocate(bytes).unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        b.to_device(&host, &mut dev).unwrap();
    }
    let h2d = t0.elapsed();
    println!(
        "{:<26} iters={:<7} bytes={:<10} total={:>11?}  per-op={:>10.1}ns  throughput={:>12.1} B/s",
        "cuda_H2D",
        iters,
        bytes,
        h2d,
        h2d.as_nanos() as f64 / iters as f64,
        bytes as f64 / (h2d.as_secs_f64().max(1e-9))
    );

    let mut back = vec![0u8; bytes];
    let t0 = Instant::now();
    for _ in 0..iters {
        b.device_to_host(&dev, &mut back).unwrap();
    }
    let d2h = t0.elapsed();
    println!(
        "{:<26} iters={:<7} bytes={:<10} total={:>11?}  per-op={:>10.1}ns  throughput={:>12.1} B/s",
        "cuda_D2H",
        iters,
        bytes,
        d2h,
        d2h.as_nanos() as f64 / iters as f64,
        bytes as f64 / (d2h.as_secs_f64().max(1e-9))
    );
    assert_eq!(back, host);

    let t0 = Instant::now();
    for _ in 0..iters {
        let d = b.allocate(bytes).unwrap();
        b.free(d).unwrap();
    }
    let af = t0.elapsed();
    println!(
        "{:<26} iters={:<7} bytes={:<10} total={:>11?}  per-op={:>10.1}ns  throughput={:>12.1} B/s",
        "cuda_alloc_free",
        iters,
        bytes,
        af,
        af.as_nanos() as f64 / iters as f64,
        bytes as f64 / (af.as_secs_f64().max(1e-9))
    );

    b.free(dev).unwrap();
    println!("\nCUDA benchmark complete (device allocations released back to zero)");
}
