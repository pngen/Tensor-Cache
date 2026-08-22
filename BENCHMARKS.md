# Benchmarks

`Tensor Cache` ships a real release-mode benchmark suite (custom harness, no
external crate) that measures the operations that matter. Run them plainly:

```
cargo bench --bench runtime --package tensorcache
cargo bench --bench cuda --package tensorcache-cuda
```

## Environment

- CPU: Intel Core (host) — performance varies by machine.
- GPU: **NVIDIA GeForce RTX 5090** (32 GiB, Blackwell).
- CUDA: driver 610.88 / CUDA UMD 13.3; runtime library loaded dynamically.
- Rust: stable 1.93.0 (MSVC), release profile (opt-level 3, lto thin).

These numbers are reported for the machine they were measured on. They are not a
claim of portability; the cost model is explicit config and must be tuned to the
deployment hardware.

## Runtime bench (host tier)

Measured with `cargo bench --bench runtime --package tensorcache -- 500 131072`
(500 iterations, 128 KiB per operation, release).

```
create_register            per-op 1.11 ms    (chunk+hash+admission+arena)
exact_lookup_hit           per-op 926 ns
cache_miss                 per-op 518 ns
compat_check(compat_id)    per-op 272 ns
host_promote_to_accel      per-op 368 us    (host -> CPU device buffer)
accel_demote_to_host       per-op 6.45 us   (device -> host, release)
persist_to_storage         per-op 3.69 ms   (durable write + manifest)
restore_from_storage       per-op 277 us
integrity_verify           per-op 965 us
admission_decision         per-op 3.8 ns
eviction_order             per-op 96 ns
planner_decision           per-op 132 ns
```

Note: `create_register` and `persist_to_storage` are dominated by SHA-256 over
the payload and filesystem writes; the de-duplication hash and CRC are per-block.

## CUDA bench

Measured with `cargo bench --bench cuda --package tensorcache-cuda -- 1048576 200`
(1 MiB transfers, 200 iterations, release) on the NVIDIA GeForce RTX 5090.

```
cuda_H2D       per-op 96 us     (~10.9 GB/s)
cuda_D2H       per-op 101 us    (~10.5 GB/s)
cuda_alloc_free per-op 1.4 us
```

The H2D/D2H throughput reflects a single 1 MiB `cudaMemcpy` plus per-call
overhead at these small sizes; larger transfers amortize the fix overhead.
Round-trip integrity was verified byte-for-byte, and device allocations returned
to zero after the benchmark (no leaked device memory).

## Honesty note

The benchmark environment and the exact command are documented here. The numbers
are real on this hardware but are not portable claims; a different platform,
backend, or block size will yield different results.
