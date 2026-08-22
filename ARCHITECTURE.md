# Architecture

`Tensor Cache` is organized as a strict systems layer for reusable tensor-shaped
state. It is a Rust workspace with three crates (`tensorcache` core,
`tensorcache-cuda` optional FFI, `tensorcache-cli` binary) and a bounded,
zero-dependency core.

## Layer diagram

```
+-----------------------------------------------------------+
|  tensorcache-cli   (create/lookup/... / coordinator / node)|
+-----------------------------------------------------------+
|  tensorcache-cuda  (optional, isolated unsafe FFI)         |
+-----------------------------------------------------------+
|  tensorcache  (core, #![forbid(unsafe_code)], zero crates) |
|   identity | compat | geometry | storage | tiers | cost    |
|   planner | admission | eviction | accounting | residency  |
|   persistence | authority | protocol | runtime | node      |
|   coordinator                                             |
+-----------------------------------------------------------+
```

## Core modules

- `identity` — `Address` (namespace/key/generation) and `ObjectId` (SHA-256 of
  the canonical address). Stable identity independent of physical location.
- `compat` — `CompatKey` and its versioned, length-prefixed canonical encoding,
  hashed to a `compat_id`. A difference in any field changes the identity, so
  a false reuse is impossible.
- `geometry` / `dtype` — validated shapes (rank/dimension bounds, checked
  products), layouts (row-major/column-major/strided), element types,
  endianness, quant identity, mutability.
- `storage` — block model, content addressing (SHA-256), CRC-32C integrity, a
  content-addressed `BlockArena` with reference counting, and safe
  reconstruction with gap/overlap/overflow validation.
- `tiers` / `residency` — `Tier` (`Accelerator`, `Host`, `Persistent`),
  `Residency`, and a move-flags state machine.
- `cost` / `planner` — a deterministic nanosecond cost model and a planner that
  chooses reuse-in-place vs transfer vs restore vs reconstruct vs reject.
- `admission` / `eviction` / `accounting` — bounded admission, deterministic
  eviction via explicit keep-score, and exact per-tier byte accounting with
  reservations and no underflow/overshoot.
- `persistence` — atomic manifest + block store, SHA-256 anchors, CRC-32C block
  checksums, and recovery that skips corrupt records without inventing state.
- `authority` — epoch, boot identity, leases, fence tokens.
- `protocol` — a bounded framed TCP protocol (magic, version, CRC, partial-read
  safe) and typed messages.
- `runtime` — the thread-safe local `TensorCache` facade (single mutex; no
  re-entrant locking).
- `coordinator` / `node` — distributed authority and storage-node logic,
  driven as independent OS processes by the CLI.

## Concurrency model

The local runtime is guarded by a single `Mutex<State>`. Internal helpers never
re-acquire the lock; a method snapshots the data it needs, releases any borrow,
and only then mutates. Mutex poisoning is recovered. This removes
read-to-write self-deadlocks, write-guard re-entry, joins-while-held, and
channel waits under lock, because there is exactly one lock and no operation
calls back into it while held.

## Distributed model

A `coordinator` process is the single source of authority (monotonic epoch,
immutable boot identity, node registry, object ownership, fence bumps, durable
snapshot). `node` processes each hold a local `TensorCache` and register with
the coordinator. Peers transfer objects directly over the framed protocol; the
coordinator gates ownership, leases, and migration. A coordinator restart
advances the epoch and rotates the boot identity (invalidating all leases) while
preserving object ownership from the snapshot.

## Boundary invariants

- A false reuse hit is forbidden: reuse requires an exact `compat_id` match.
- A migration never creates dual authoritative owners: the fence is bumped and
  the old owner's authority becomes stale.
- Authority is never committed before verification.
- Accounting never underflows or overshoots; placeholders are not invented.
- No phantom objects, no invented commits, no stale authority.
