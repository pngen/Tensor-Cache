#![allow(unsafe_code)]
//! Optional CUDA backend for Tensor Cache.
//!
//! This crate is the only place where unsafe FFI is permitted. It loads the
//! CUDA runtime library dynamically at runtime so that the core does not
//! carry a link-time CUDA dependency.

// Placeholder backend; implemented in a later build step.
