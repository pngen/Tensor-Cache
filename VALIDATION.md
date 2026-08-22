# Validation

This documents what was actually validated on the build machine, and the honest
limitations.

## Toolchain

- Rust stable 1.93.0 (x86_64-pc-windows-msvc), rustfmt 1.8.0, clippy 0.1.93.
- NVIDIA GeForce RTX 5090 (32 GiB, Blackwell), driver 610.88, CUDA UMD 13.3,
  CUDA toolkit 12.9/13.x (runtime library loaded dynamically).
- No external crates are used anywhere in the workspace.

## Commands run plainly (no timeouts)

- `cargo fmt --check` : passes.
- `cargo build` : clean debug build, zero warnings.
- `cargo build --release` : clean release build, zero warnings.
- `cargo build --features cuda -p tensorcache-cli` : builds the CUDA feature.
- `cargo build --examples --workspace` : all examples build; zero warnings.
- `cargo test --workspace` : **96 tests pass**, 0 failed, 0 ignored.
- `cargo test -p tensorcache-cuda` : 2 real CUDA tests pass on the RTX 5090.
- `cargo test -p tensorcache-cuda --test cuda_runtime` : 1 runtime+CUDA test
  passes on the RTX 5090.
- `cargo test -p tensorcache-cli --test distributed` : 1 real multi-process
  coordinator/node test passes (create, distributed transfer/replica,
  migration, stale-owner rejection, coordinator restart w/ snapshot).
- `cargo clippy --workspace --all-targets -- -D warnings` : passes.
- `cargo clippy -p tensorcache-cli --features cuda --all-targets -- -D warnings`
  : passes.
- Examples run: `cache_hit`, `compat_rejection`, `tiered_residency`,
  `capacity_pressure`, `crash_recovery`, `dedup`, `reconstruction_economics`,
  `distributed` (real processes), `cuda_accelerator` (real CUDA).
- Benchmarks run (release): `runtime` bench and `cuda` bench on the RTX 5090 —
  see `BENCHMARKS.md`.

## What is validated

- SHA-256 and CRC-32C against known vectors.
- Canonical compatibility encoding and rejection of geometry/dtype/layout/model/
  revision/runtime/operation differences (no false reuse).
- Shape/rank/dimension/overflow protection, stride/layout validation.
- Block chunking/reconstruction with gap/overlap/corruption detection.
- Deduplication sharing physical bytes without identity collapse.
- Admission, deterministic eviction, and exact resource accounting (no
  underflow / overshoot).
- Residency transitions (promote/demote/persist/restore/evict).
- Atomic persistence, SHA-256 manifest anchors, CRC-32C block checksums,
  truncated/corrupt manifest rejection, temp-file cleanup, block corruption
  rejection, path-traversal rejection.
- Framed protocol: magic/version/length/CRC and malformed-frame rejection,
  partial-read safety, typed-message round trips.
- Authority: epoch monotonicity, boot identity, expired lease, stale fence,
  stale epoch/boot, migration fence bump and stale-owner rejection after
  migration.
- Real multi-process coordinator + two nodes: object create, distributed
  transfer, replica, migration, stale owner rejection, coordinator restart
  preserving ownership from the snapshot.
- Real CUDA on the NVIDIA RTX 5090: device discovery, allocation, H2D, D2H,
  integrity after round-trip, fill, free, repeated allocation/free, capacity
  accounting, no leaked device allocations, and the full runtime
  promote/restore/verify/demote accelerator flow.

## Repeated runs

The full `cargo test --workspace` and the distributed test were re-run multiple
times during hardening; results were stable (no stale-state dependence).

## Deferred toolchain mitigations

Sanitizers (AddressSanitizer / ThreadSanitizer) are **not available** under the
installed stable Rust toolchain on this platform, and per policy the developer
environment was not mutated to add them. The mitigation instead relies on:

- `#![forbid(unsafe_code)]` in the core (the only unsafe code is the isolated
  CUDA FFI),
- overflow checks and strict bounds throughout,
- debug assertions,
- hostile-input tests (malformed frames, truncated manifests, corrupt blocks,
  path traversal),
- concurrency tests (single-mutex runtime),
- real multi-process tests,
- repeated lifecycle runs.

## Known limitations (honest)

- The CUDA backend is real and validated **only** where a CUDA device was
  present (the RTX 5090 on this machine). On a host without CUDA, the
  `cuda` feature still compiles but the backend reports unavailable and tests
  skip.
- The planner cost model is explicit configuration and is not a claim of
  portable microbenchmarks.
- QUIC transport is not implemented (bounded framed TCP is used for 1.0.0).
- HIP, Level Zero, Metal, and Vulkan are not implemented (the backend contract
  accommodates them); they are not claimed as present.
- The coordinator snapshot is persisted, but node re-registration after a
  coordinator restart requires the application to re-establish leases with the
  new epoch/boot identity.
