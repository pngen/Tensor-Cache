#![allow(unsafe_code)]
//! Optional CUDA backend for Tensor Cache (isolated unsafe FFI).
//!
//! The CUDA runtime API is loaded dynamically at runtime so the core never
//! carries a link-time CUDA dependency. See backend.rs for the backend and
//! loader.rs for the dynamic loader.

mod backend;
mod loader;

pub use backend::CudaBackend;
