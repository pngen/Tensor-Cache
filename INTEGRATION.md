# Integration

`Tensor Cache` is a Rust library and a binary. Integrate it either as a library
(in-process runtime) or as a distributed runtime (coordinator + node processes).

## Library (in-process runtime)

Add `tensorcache` to your `Cargo.toml` and use the `TensorCache` facade:

```
use tensorcache::runtime::{TensorCache, RuntimeConfig};
use tensorcache::compat::CompatKey;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};

let config = RuntimeConfig { host_capacity: 1 << 30, ..Default::default() };
let tc = TensorCache::new(config)?;

let compat = CompatKey {
    dtype: Dtype::F32,
    shape: Shape::new(vec![128, 4096])?,
    layout: Layout::RowMajor,
    model: Some("my-model".into()),
    ..Default::default()
};
let oid = tc.register("my-namespace", "layer3.out", 0, compat.clone(), &payload)?;
let res = tc.lookup("my-namespace", "layer3.out", 0, &compat)?;
let bytes = tc.restore(&oid, &Tier::Host)?;
tc.verify(&oid)?;
```

To register an accelerator backend (e.g. CUDA), construct the backend and pass it
to `TensorCache::with_backends`:

```
let cuda = tensorcache_cuda::CudaBackend::new(0, 1 << 30)?;
let tc = TensorCache::with_backends(config, vec![Box::new(cuda)])?;
```

## Distributed runtime

Run a coordinator and one or more nodes as independent OS processes:

```
target/debug/tensorcache coordinator --listen 127.0.0.1:9000 --snapshot ./coord.snap
target/debug/tensorcache node --id n1 --listen 127.0.0.1:9001 --coordinator 127.0.0.1:9000 --store ./store-a
target/debug/tensorcache node --id n2 --listen 127.0.0.1:9002 --coordinator 127.0.0.1:9000 --store ./store-b
```

The nodes register with the coordinator for authority, serve peer fetches, store
replicas, and participate in coordinator-authorized migration. See
`PROTOCOL.md` for the wire format and `RECOVERY.md` for restart semantics.

## Re-associating durable state

On restart, `TensorCache::new` recovers durable objects into the persistent
placement. Only objects whose blocks all exist on disk are admitted; skipped
objects are counted in the recovery report. The application re-associates any
caller-owned semantic meaning through the `CompatKey` identity strings it
supplied (see `PERSISTENCE.md`).

## Building the CUDA feature

```
cargo build -p tensorcache-cli --features cuda
cargo test -p tensorcache-cuda --features cuda
```

The `cuda` feature is optional and loads the CUDA runtime dynamically; the core
has no link-time CUDA dependency.
