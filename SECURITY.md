# Security

`Tensor Cache` is designed to be hostile-input aware and to avoid executing
arbitrary code, loading plugins, evaluating scripts, shelling out to remote data,
or unsafe deserialization.

## Parsing and protocol

- The framed protocol has an explicit magic/version; **bad magic or version is
  rejected** before any payload is read.
- Frame length is bounded by `MAX_FRAME` (64 MiB) and **rejected before
  allocation**. The reader never allocates based on a peer-controlled length
  beyond the cap.
- Length-prefixed strings/blobs validate the length against the remaining input.
- Invalid boolean bytes and unknown message types are rejected.
- A frame CRC mismatch or truncated frame abandons the connection.

## Geometry and storage

- `Shape` rejects rank abuse (> 32) and per-dimension overflow; product and
  byte-length are checked and overflow before allocation.
- `Layout` validates strides (nonzero, no offset overflow).
- Block lists are validated (contiguous offsets, bounded count, no
  duplicate/out-of-range blocks) before reconstruction allocates.
- Block \*\*and manifest integrity use SHA-256 anchors and CRC-32C block
  checksums; corrupt data is rejected, not silently accepted.

## Path traversal

Block and manifest file names must be exactly 64 hex characters; anything else is
rejected. No user-supplied path is used directly.

## Authority

- Stale epoch / boot identity / expired lease / stale fence are rejected.
- Migration bumps the fence so the old owner cannot mutate; no dual
  authoritative owners.
- A restarted node never silently inherits old authority.

## Allocation bombs and impossible values

- Frame lengths, tensor ranks, shape products, block counts, stride products and
  byte lengths are all bounded/checked before allocation.
- Duplicate conflicting identities (`ObjectId` collisions from the same address,
  or an already-registered object) are rejected with `Exists`.

## No dynamic execution

- No arbitrary code execution, no plugin loading, no script evaluation, no shell
  invocation from remote data, no unsafe deserialization (the only unsafe code is
  the isolated CUDA FFI, which is limited to fixed device-memory operations).

## Known mitigation notes

- The CUDA FFI is the only unsafe code, sitting in the `tensorcache-cuda` crate
  behind the `cuda` feature, and callable only through the narrow `Backend`
  contract.
- The toolchain does not provide sanitizers on this platform; the core uses
  `#![forbid(unsafe_code)]`, overflow checks, strict bounds, debug assertions,
  hostile-input tests, concurrency tests and multi-process tests instead. See
  `VALIDATION.md`.
