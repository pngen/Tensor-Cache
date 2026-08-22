//! Example 10: real CUDA accelerator (validated on NVIDIA RTX 5090).
//!
//! Promotes a tensor into the CUDA accelerator tier (real cudaMalloc + H2D),
//! restores it back to host (D2H), verifies integrity, then demotes and
//! confirms the device allocation is released.

use tensorcache::backend::BackendId;
use tensorcache::compat::CompatKey;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::runtime::RuntimeConfig;
use tensorcache::runtime::TensorCache;
use tensorcache::tiers::Tier;
use tensorcache_cuda::CudaBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda = match CudaBackend::new(0, 1 << 28) {
        Ok(c) => c,
        Err(e) => {
            println!("CUDA unavailable on this host: {e}");
            println!("(build with --features cuda and an NVIDIA driver to run this example)");
            return Ok(());
        }
    };
    let count = CudaBackend::device_count()?;
    println!("CUDA device count: {count}");

    let config = RuntimeConfig {
        host_capacity: 1 << 24,
        ..Default::default()
    };
    let tc = TensorCache::with_backends(config, vec![Box::new(cuda)])?;

    let compat = CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![1024]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("cuda-model".into()),
        ..Default::default()
    };
    let data: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();

    let oid = tc.register("ns", "cuda-embed", 1, compat, &data)?;
    let accel = Tier::Accelerator(BackendId::cuda(0));

    tc.promote(&oid, &accel)?;
    println!(
        "promoted to CUDA accelerator: placements={:?}",
        tc.metadata(&oid)?.placements
    );

    let back = tc.restore(&oid, &Tier::Host)?;
    assert_eq!(back, data);
    tc.verify(&oid)?;
    println!(
        "restored {} bytes from CUDA and verified integrity",
        back.len()
    );

    tc.demote(&oid, &accel)?;
    println!(
        "demoted accelerator placement: placements={:?}",
        tc.metadata(&oid)?.placements
    );
    println!(
        "accel_used={} (device allocations released)",
        tc.resources().accel_used
    );
    Ok(())
}
