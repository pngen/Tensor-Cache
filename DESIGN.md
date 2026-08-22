# Design

This document records the key design decisions and the reasons behind them.

## Zero external dependencies in the core

The core (`tensorcache`) uses only the Rust standard library. SHA-256 and CRC-32C
are hand-implemented and validated against known vectors (the canonical check
values), so there is no transitive dependency surface and no hidden unsafe code
pulled in by a dependency. The CUDA backend is isolated in a separate crate and
loaded dynamically at runtime.

## Canonical compatibility identity

Reuse is gated by a `CompatKey` (model, revision, runtime, operation, dtype,
shape, layout, endianness, device, precision, quant, mutability) encoded with a
leading version byte and length-prefixed strings/byte blobs. The encoding is
versioned and unambiguous — no delimiter-collision tricks. A SHA-256 digest of
the canonical bytes is the compatibility identity; a difference in any field
changes it. This makes accidental reuse across geometry, dtype, layout, model,
runtime, revision, quantization, or semantic meaning impossible.

## Safe reuse lookup

A request carries an `ObjectId` (namespace/key/generation). `lookup` returns a
hit only if the stored entry's `compat_id` equals the request's. A present but
incompatible entry is rejected with a `Compatibility` error (not silently
served), and a present compatible entry reports its source tier, bytes reused,
reconstruction avoided, and transfer cost.

## Deduplication without identity collapse

Physical bytes are content-addressed. A `BlockArena` deduplicates identical
blocks and tracks reference counts. Two logical objects may share physical
bytes, but their `ObjectId`, `CompatKey`, generation, lifecycle, and ownership
remain distinct. Mutable objects never silently share storage in a way that
permits cross-object mutation, because mutation is gated by the fence/lease and
isolated per placement.

## Deterministic economics

Costs are integer nanoseconds from a `CostModel`. The `planner` compares
reuse-in-place, transfer-from-a-tier, restore-from-storage, and reconstruct, and
chooses the lowest cost with a stable tie-break. The cost model is explicit
config, not a claim of portability; operators tune it to observed hardware.

## Admission and eviction

Admission is bounded: it rejects an object that cannot be satisfied by the
current tier budget, evicting the lowest-scoring entries first to make room. The
eviction keep-score weighs reuse count, recency, reconstruction cost, size,
priority, durability, and tier pressure; ordering is deterministic (ties broken
by object id). No HashMap iteration order is used.

## Persistence atomicity

A manifest is written to a temp file, flushed, then renamed into place. It is
anchored by a trailing SHA-256 over the manifest body, and each block file is
verified against its content hash and a CRC-32C block checksum. Recovery skips
corrupt/truncated records and never invents committed state.

## Authority

A coordinator epoch advances on restart, a boot identity rotates, and object
ownership carries a monotonic fence. A mutation requires a current lease (not
expired) with a matching epoch/boot and a fence at least the object's current
fence. Migration bumps the fence, so the old owner cannot mutate afterward.

## Why a single mutex in the local runtime

For 1.0.0 the local runtime uses a single mutex for correctness and simplicity.
The distributed runtime is what scales out (independent processes); the local
runtime emphasizes correctness and the absence of deadlocks. The single lock
eliminates lock-ordering and re-entrancy classes entirely, which is the highest
value for a systems substrate whose invariants are critical.
