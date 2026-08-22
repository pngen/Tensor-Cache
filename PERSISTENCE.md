# Persistence

Durable tensor persistence uses atomic commit semantics and content-addressed
block storage.

## On-disk layout

```
<root>/blocks/<content_hash_hex>     one file per unique block
<root>/manifests/<object_id_hex>.manifest   one versioned manifest per tensor
```

Block filenames are exactly 64 hex characters derived from the SHA-256 of the
block bytes; any other string is rejected (no path traversal).

## Block files

A block is content-addressed. `put_block` writes a temp file, flushes it, then
renames it into place; if the file already exists with matching bytes it is a
no-op, and if it exists with different bytes the on-disk data is corrupt and an
error is raised. `get_block` reads the file, recomputes its SHA-256, and rejects
a mismatch.

## Manifests

A manifest embeds the durable metadata needed to re-associate a tensor after a
crash: the object id (address), the canonical `CompatKey`, `byte_len`,
`numel`, creation time, and the ordered block list (content hash, offset,
length, CRC-32C). A 32-byte **SHA-256 anchor** over the manifest body is
appended; a manifest is rejected if the anchor does not match.

Commit = write a temp sibling file, `sync_all`, then rename into place (atomic
replace where the platform supports it). Leftover temp/backup files are cleaned
up on open and are never treated as committed state.

## Durable vs caller-owned metadata

**Durable by Tensor Cache:** object id (namespace/key/generation), compatible
identity (geometry/dtype/layout/model/revision/runtime/operation/precision/
quant/mutability), byte length, element count, creation time, and the integrity
block list.

**Caller-owned (must be re-associated by the application):** the raw tensor
payload is reconstructed from the durable blocks on demand; any application-level
semantic meaning of the tensor (e.g. what computation produced it) is carried by
the `CompatKey` identity strings that the caller supplied. Nothing else about
the producing computation is persisted.
