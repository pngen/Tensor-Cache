//! Shared helpers for Tensor Cache examples.

#![allow(dead_code)]

use std::path::PathBuf;

use tensorcache::admission::AdmissionPolicy;
use tensorcache::compat::CompatKey;
use tensorcache::cost::CostModel;
use tensorcache::dtype::Dtype;
use tensorcache::eviction::EvictionPolicy;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::runtime::RuntimeConfig;

pub fn compat(dtype_tag: Dtype, dims: Vec<u64>, model: &str) -> CompatKey {
    CompatKey {
        dtype: dtype_tag,
        shape: Shape::new(dims).unwrap(),
        layout: Layout::RowMajor,
        model: Some(model.to_string()),
        ..Default::default()
    }
}

pub fn startup_cost() -> CostModel {
    CostModel::default()
}

pub fn config(host_bytes: u64, persistent: Option<PathBuf>) -> RuntimeConfig {
    RuntimeConfig {
        host_capacity: host_bytes,
        persistent_path: persistent,
        block_size: 1024,
        admission: AdmissionPolicy::default(),
        eviction: EvictionPolicy::default(),
        cost: CostModel::default(),
        ..Default::default()
    }
}

pub fn payload(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8).collect()
}

pub fn temp_dir(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::env::temp_dir().join(format!("tc-example-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

pub fn f32(dims: Vec<u64>, model: &str) -> CompatKey {
    compat(Dtype::F32, dims, model)
}
