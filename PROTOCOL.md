# Protocol

`Tensor Cache` uses a bounded, framed protocol over TCP for the 1.0.0 wire format
(the coordinator protocol and the peer protocol share one codec). QUIC is
documented as future work and is not implemented.

## Frame layout

Every frame is a fixed 16-byte header followed by a payload.

```
offset  size  field
0       4     magic  u32 little-endian, value 0x54434246 ("TCBF")
4       1     version u8,  value 1
5       1     message-type tag u8
6       4     length u32 little-endian  (payload byte count)
10      4     crc    u32 little-endian  (CRC-32C over header bytes 0..10 + payload)
14      2     reserved (zero)
```

The reader never allocates based on a peer-controlled length beyond a hard cap
(`MAX_FRAME` = 64 MiB); larger frames are rejected before allocation. Bad magic,
bad version, an unknown message type, a truncated frame, an over-cap length, or a
CRC mismatch all yield a `Protocol` error and the connection is abandoned.

## Message types

```
1  Register      node_id, addr
2  Hello         epoch, boot_id, node_id, addr, lease_ns
3  Lookup        namespace, key, generation, compat
4  LookupResult  found, generation, owner?, owner_addr?
5  Create        namespace, key, generation, byte_len, compat, node_id
6  CreateAck     object_id, epoch, fence, owner
7  LeaseRenew    object_id, fence
8  LeaseGrant    object_id, epoch, fence, expires_ns
9  Fetch         object_id, compat
10 FetchReply    object_id, data, crc
11 Store         namespace, key, generation, data, crc, compat, source
12 StoreAck      object_id
13 Migrate       object_id, new_owner, new_owner_addr, fence
14 MigrateAck    object_id, new_owner, fence
15 Heartbeat     node_id, epoch
16 Error         code, message
```

## Primitive encoding

- `u8` / `u16` / `u32` / `u64` / `i64`: fixed-width little-endian.
- `bool`: one byte (0 or 1); any other value is rejected.
- `str`: a `u64` length prefix followed by UTF-8 bytes.
- `bytes`: a `u64` length prefix followed by raw bytes.

Length prefixes are validated against the remaining input before any
allocation, so a peer cannot force a huge allocation.

## Transport semantics

- The coordinator and each node listen for incoming TCP connections.
- Each connection is served by its own thread; the server reads one frame,
  dispatches it, writes zero or more response frames, and continues until the
  peer closes or a protocol error occurs.
- Node-to-node object transfer (Fetch / Store) and coordinator-driven migration
  reuse this framing.

## Authority in the protocol

The coordinator speaks the authority protocol. It issues `Hello` (epoch and
boot identity) on registration, `CreateAck` (ownership fence), `LeaseGrant`
for renewals, and `Migrate` instructions. Stale epoch, stale boot identity,
expired lease, or a fence below the object's current fence are rejected with an
`Error` carrying code `authority`.
