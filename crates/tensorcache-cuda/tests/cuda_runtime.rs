//! Runtime + real CUDA integration: promote/restore/verify/demote a tensor
//! through the accelerator (CUDA) tier.

use tensorcache::backend::BackendId;
use tensorcache::compat::CompatKey;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::runtime::{RuntimeConfig, TensorCache};
use tensorcache::tiers::Tier;
use tensorcache_cuda::CudaBackend;

#[test]
fn runtime_promote_restore_verify_demote_to_cuda() {
    let cuda = match CudaBackend::new(0, 1 << 28) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("CUDA unavailable ({e}); skipping accelerator integration validation");
            return;
        }
    };
    let config = RuntimeConfig {
        host_capacity: 1 << 20,
        ..Default::default()
    };
    let tc = TensorCache::with_backends(config, vec![Box::new(cuda)]).unwrap();

    let compat = CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![256]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("model-a".into()),
        ..Default::default()
    };
    let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let oid = tc.register("ns", "accel", 1, compat, &data).unwrap();

    // Promote to the CUDA accelerator tier (real cudaMalloc + H2D).
    let accel = Tier::Accelerator(BackendId::cuda(0));
    tc.promote(&oid, &accel).unwrap();
    assert!(tc.metadata(&oid).unwrap().placements.contains(&accel));
    assert_eq!(tc.resources().accel_used, 1024);

    // Restore (reads the device back to host via D2H) and verify bytes.
    let bytes = tc.restore(&oid, &Tier::Host).unwrap();
    assert_eq!(bytes, data);
    tc.verify(&oid).unwrap();

    // Demote the accelerator placement: device buffer is freed, accounting
    // returns to zero.
    tc.demote(&oid, &accel).unwrap();
    assert!(!tc.metadata(&oid).unwrap().placements.contains(&accel));
    assert_eq!(tc.resources().accel_used, 0);
}
