//! Tensor Cache: an open-source, vendor-neutral distributed runtime for
//! caching and governing reusable tensor-shaped computational state.

#![forbid(unsafe_code)]

pub mod accounting;
pub mod admission;
pub mod authority;
pub mod backend;
pub mod backend_cpu;
pub mod compat;
pub mod coordinator;
pub mod cost;
pub mod crc;
pub mod dtype;
pub mod error;
pub mod eviction;
pub mod geometry;
pub mod hash;
pub mod ident;
pub mod node;
pub mod persistence;
pub mod planner;
pub mod protocol;
pub mod residency;
pub mod runtime;
pub mod storage;
pub mod tiers;
pub mod wire;
