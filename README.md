# Tensor Cache 1.0.0

`Tensor Cache` is an open-source, vendor-neutral distributed runtime for caching
and governing reusable tensor-shaped computational state across accelerator-local
memory, system memory, persistent storage, processes, and execution nodes.

Its core systems question is:

> **Where should reusable tensor state live, when should it move, when should it
> be reused, and when is reconstruction cheaper than retention or transfer?**

Modern AI systems produce expensive intermediate tensors — embeddings, layer
activations, attention intermediates, recurrent state, encoder outputs,
speculative-decoding intermediates, adapter-derived artifacts — that may be
reusable but compete for accelerator capacity. `Tensor Cache` makes their
identity, compatibility, residency, reuse, movement, eviction, persistence,
recovery, integrity, and economics **explicit** and correct.

## What Tensor Cache is not

`Tensor Cache` is **not** merely a hash map containing tensors. It is not an LRU
cache. It is not a tensor dictionary. It is not a generic object store. It is
not tied to any inference engine, framework, accelerator vendor, or model.

It is a systems runtime for tensor cache **identity**, **compatibility**,
**placement**, **residency**, **reuse**, **movement**, **eviction**,
**reconstruction economics**, **integrity**, **persistence**, **replication**,
**recovery**, **distributed authority**, and **lifecycle**.

## Architectural boundary

`Tensor Cache` deliberately leaves the following to the existing public
infrastructure stack and does **not** duplicate them:

- **FlashTier** manages where bytes reside across heterogeneous memory tiers.
- **Context Fabric** manages arbitrary reusable computational state.
- **Compute Fabric** manages where computation executes.
- **Reclaim Fabric** manages whether accumulated machine state remains worth
  keeping.
- **Checkpoint Fabric** manages what execution state survives.
- **KV Fabric** manages reusable KV / prefix inference state.

`Tensor Cache` owns the specialized cache substrate for **reusable
tensor-shaped state**: tensor identity, geometry, dtype, layout, device and
runtime/model compatibility, chunking, immutable entries and mutable-version
semantics, admission, lookup, exact compatibility, reuse, residency, tier
movement, replication, safe deduplication, eviction, persistence,
reconstruction/transfer cost, integrity, distributed ownership and authority,
crash recovery, and capacity governance.

## Supported tiers

- **Accelerator tier** — device-resident buffers. The mandatory `cpu` backend
  is always present; an optional real `cuda` backend is provided behind
  `--features cuda` and is loaded dynamically (no link-time CUDA dependency).
  HIP, Level Zero, Metal, and Vulkan are valid future backends but are not
  claimed as implemented today.
- **Host-memory tier** — a content-addressed block arena with deduplication.
- **Persistent-storage tier** — durable, atomic, manifest+bold-block storage
  with SHA-256 anchors and CRC-32C block checksums.

## Quick start

### Build

```
cargo build
cargo test
```

Requirements: a stable Rust toolchain (1.75+), no external crates, no network,
no telemetry, no account system. Fully local operation is supported.

### CLI

```
target/debug/tensorcache create --store ./store --namespace demo --key emb \
    --dtype f32 --shape 16x16 --fill 4096
target/debug/tensorcache lookup --store ./store --namespace demo --key emb \
    --dtype f32 --shape 16x16
target/debug/tensorcache stats --store ./store
```

### Run the examples

```
cargo run --example cache_hit --package tensorcache
cargo run --example compatibility_rejection --package tensorcache
cargo run --example tiered_residency --package tensorcache
cargo run --example capacity_pressure --package tensorcache
cargo run --example crash_recovery --package tensorcache
cargo run --example dedup --package tensorcache
cargo run --example reconstruction_economics --package tensorcache
cargo run --example distributed --package tensorcache-cli   # needs the CLI built
cargo run --example cuda_accelerator --package tensorcache-cuda  # with CUDA
```

## The library

The public facade is `tensorcache::runtime::TensorCache`:

```
use tensorcache::runtime::TensorCache;
use tensorcache::compat::CompatKey;

let tc = TensorCache::new(Default::default())?;
let compat = CompatKey { ..., ..Default::default() };
let oid = tc.register("ns", "key", 0, compat.clone(), &payload)?;
let res = tc.lookup("ns", "key", 0, &compat)?;   // compatible reuse
tc.persist(&oid)?;                               // durable copy
let bytes = tc.restore(&oid, &Tier::Host)?;      // materialize + return
tc.verify(&oid)?;                                // integrity across placements
```

## Testing

All 96 tests across the workspace pass, including a real multi-process
coordinator/node integration test and real CUDA validation on an NVIDIA GeForce
RTX 5090. Run everything plainly (no timeouts):

```
cargo test --workspace
```

## Limitations

See `VALIDATION.md` for the complete, honest account. Key points: the CUDA
backend is real and validated only where a CUDA device was present (this
machine); the `cuda` feature is optional; the planner cost model is config
input and is not a claim of portable microbenchmarks; QUIC transport is future
work (bounded framed TCP is used for 1.0.0).

## Roadmap

- HIP / Level Zero / Metal / Vulkan backends (backend contract already
  accommodates them).
- QUIC as an alternative transport.
- Coordinator durable snapshot already implemented; richer reconciliation.
- Namespace/tenant quota enforcement at the coordinator.
- Content-addressed structural reuse across namespaces under explicit opt-in.

## Relation to the fabric stack

- **FlashTier** manages where bytes reside.
- **Context Fabric** manages arbitrary reusable computational state.
- **Compute Fabric** manages where computation runs.
- **Reclaim Fabric** manages whether state remains worth retaining.
- **Checkpoint Fabric** manages what execution state survives.
- **KV Fabric** manages reusable KV / prefix inference state.
- **Tensor Cache** manages specialized reusable tensor-cache state.

`Tensor Cache` may be conceptually governed by `Context Fabric` as reusable
computational state, but it owns the specialized mechanics and economics of
caching tensor-shaped artifacts. It is narrower than `KV Fabric` (which
specializes in KV/prefix reusable state) and must not be reduced to a rename of
it.

## License

Apache-2.0. See `LICENSE`.
