# Recovery

Recovery is a first-class feature. It must converge to a coherent state without
inventing commits or admitting phantom objects.

## Persistent-state recovery

On opening a store, `PersistentStore::recover` scans the manifest directory. For
each record:

1. It reads the manifest body and verifies the trailing SHA-256 anchor.
2. It decodes the versioned fields and validates geometry consistency
   (`byte_len` must equal shape `*. ` dtype byte size; `numel` must equal the
   shape product) and the block list (contiguous offsets, no gaps/overlaps, no
   out-of-range, bounded count).
3. It verifies that every referenced block exists on disk.

A record that fails any step is **skipped** without failing the whole recovery.
No phantom entry is created for a corrupt record, and no commit is invented.

## Runtime recovery

`TensorCache::recover` (called from `new`) uses the persistent store's recovered
manifests to re-associate each durable object as a `Persistent` placement. Only
objects whose blocks all exist on disk are admitted; the rest are skipped and
counted. A skipped object is not served and is not counted in the recovered set.

## Coordinator restart

A coordinator restart advances the epoch and rotates the boot identity (via the
durable snapshot in `coordinator.rs`), which invalidates every prior node lease.
Object ownership is preserved from the snapshot. Nodes must re-register /
renew against the new epoch and boot identity; a node holding a stale epoch or
boot identity cannot mutate. Read-only lookups still resolve because the
coordinator retains the ownership map.

## Recovery cases exercised

- Incomplete commits (leftover temp files) — cleaned on open.
- Corrupt / truncated manifests — skipped with a warning count.
- Missing blocks — the owning object is skipped.
- Abrupt node loss — a later peer fetch fails cleanly rather than serving
  corrupt state.
- Stale epoch / boot identity — authority rejected.
- Partially migrated objects — the old owner retains coherent state and the
  coordinator only transfers ownership after a verified MigrateAck.
- No phantom entries — an object with zero placements and no durability is
  removed; a recovered object must have all blocks present.
