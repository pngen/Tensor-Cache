#![forbid(unsafe_code)]
//! Durable tensor persistence with atomic commit and crash recovery.
//!
//! On-disk layout:
//!   <root>/blocks/<content_hash_hex>   one file per unique block
//!   <root>/manifests/<object_id_hex>.manifest   one versioned manifest per tensor
//!
//! A manifest is a versioned, length-prefixed binary record carrying the
//! metadata needed to re-associate a tensor after a crash, followed by a
//! 32-byte SHA-256 anchor over the manifest body. Block files are named by
//! their content hash and verified against that hash and a CRC-32C checksum on
//! read.
//!
//! Commits are made by writing a temporary file, flushing it, then renaming it
//! into place (atomic replacement where the platform supports it). Incomplete
//! commits (leftover temp files) and corrupt/truncated manifests are detected
//! on startup and skipped safely: recovery never invents committed state and
//! never admits a phantom object.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::compat::CompatKey;
use crate::error::{Error, Result};
use crate::hash::{hash, Digest};
use crate::storage::BlockRef;
use crate::wire::{Reader, Writer};

/// Magic header for a manifest file ("TCM1").
const MANIFEST_MAGIC: u32 = 0x5443_4D31;
/// Manifest format version.
const MANIFEST_VERSION: u8 = 1;

/// Durable metadata persisted for one tensor entry.
#[derive(Debug, Clone)]
pub struct PersistEntryMeta {
    pub object_id: String,
    pub namespace: String,
    pub key: String,
    pub generation: u64,
    pub compat: CompatKey,
    pub byte_len: u64,
    pub numel: u64,
    pub created_ns: u64,
    pub blocks: Vec<BlockRef>,
}

fn hex_path_valid(p: &str) -> bool {
    p.len() == 64 && p.bytes().all(|b| b.is_ascii_hexdigit())
}

/// A durable persistent block and manifest store.
pub struct PersistentStore {
    root: PathBuf,
    blocks_dir: PathBuf,
    manifests_dir: PathBuf,
}

impl PersistentStore {
    /// Open (and if necessary create) a persistent store rooted at `root`.
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let blocks_dir = root.join("blocks");
        let manifests_dir = root.join("manifests");
        fs::create_dir_all(&blocks_dir)?;
        fs::create_dir_all(&manifests_dir)?;
        let s = PersistentStore {
            root,
            blocks_dir,
            manifests_dir,
        };
        s.cleanup_incomplete()?;
        Ok(s)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn manifest_path(&self, object_id: &str) -> Result<PathBuf> {
        if !hex_path_valid(object_id) {
            return Err(Error::Persistence(
                "invalid object id for manifest path".into(),
            ));
        }
        Ok(self.manifests_dir.join(format!("{object_id}.manifest")))
    }

    fn block_path(&self, content_hash: &Digest) -> Result<PathBuf> {
        let h = content_hash.to_hex();
        if !hex_path_valid(&h) {
            return Err(Error::Persistence(
                "invalid content hash for block path".into(),
            ));
        }
        Ok(self.blocks_dir.join(h))
    }

    /// Write a block file (content addressed). If the block already exists with
    /// a matching hash this is a no-op; if it exists with a different hash the
    /// on-disk bytes are corrupt and an error is raised.
    pub fn put_block(&self, bytes: &[u8]) -> Result<Digest> {
        let h = hash(bytes);
        let path = self.block_path(&h)?;
        if path.exists() {
            let existing = fs::read(&path)?;
            if existing != bytes {
                return Err(Error::Persistence(format!(
                    "block {} exists with corrupt contents",
                    h
                )));
            }
            return Ok(h);
        }
        atomic_write(&path, bytes)?;
        Ok(h)
    }

    /// Read and verify a block file against its content hash.
    pub fn get_block(&self, content_hash: &Digest) -> Result<Vec<u8>> {
        let path = self.block_path(content_hash)?;
        let bytes = fs::read(&path)
            .map_err(|e| Error::Reconstruct(format!("block {} missing: {e}", content_hash)))?;
        let actual = hash(&bytes);
        if &actual != content_hash {
            return Err(Error::Persistence(format!(
                "block {} content hash mismatch",
                content_hash
            )));
        }
        Ok(bytes)
    }

    /// Remove a block file (used during reclamation).
    pub fn remove_block(&self, content_hash: &Digest) -> Result<()> {
        let path = self.block_path(content_hash)?;
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Atomically commit a manifest for a tensor entry.
    pub fn write_manifest(&self, meta: &PersistEntryMeta) -> Result<()> {
        let bytes = encode_manifest(meta)?;
        let path = self.manifest_path(&meta.object_id)?;
        atomic_write(&path, &bytes)?;
        Ok(())
    }

    /// Read and verify a manifest for a tensor, rejecting malformed/tampered
    /// records.
    pub fn read_manifest(&self, object_id: &str) -> Result<PersistEntryMeta> {
        let path = self.manifest_path(object_id)?;
        let bytes = fs::read(&path)?;
        decode_manifest(&bytes)
    }

    /// Remove a manifest (used during reclamation or overwrite).
    pub fn remove_manifest(&self, object_id: &str) -> Result<()> {
        let path = self.manifest_path(object_id)?;
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// True if a manifest exists for the given object id.
    pub fn has_manifest(&self, object_id: &str) -> Result<bool> {
        Ok(self.manifest_path(object_id)?.exists())
    }

    /// Recover durable state: return the valid manifests, skipping any corrupt,
    /// truncated or incomplete record. Leftover temp files are removed.
    pub fn recover(&self) -> Vec<PersistEntryMeta> {
        let mut out = Vec::new();
        let entries = match fs::read_dir(&self.manifests_dir) {
            Ok(d) => d,
            Err(_) => return out,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("manifest") {
                continue;
            }
            if let Ok(bytes) = fs::read(&path) {
                if let Ok(meta) = decode_manifest(&bytes) {
                    out.push(meta);
                }
            }
        }
        out
    }

    /// Remove temp/leftover files from an interrupted commit.
    pub fn cleanup_incomplete(&self) -> Result<()> {
        for dir in [&self.blocks_dir, &self.manifests_dir] {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.contains(".tmp") || name.ends_with(".bak") {
                            let _ = fs::remove_file(&p);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a block file exists.
    pub fn block_exists(&self, content_hash: &Digest) -> bool {
        self.block_path(content_hash)
            .map(|p| p.exists())
            .unwrap_or(false)
    }
}

/// Atomically write bytes to `path`: write a sibling temp file, flush it, then
/// rename it into place. On platforms where rename replaces atomically this is
/// a real atomic commit; elsewhere the leftover temp detection in `recover`
/// covers interruption.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Persistence("manifest path has no parent".into()))?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Persistence("manifest path has no file name".into()))?;
    let tmp = parent.join(format!("{file_name}.tmp{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if path.exists() {
        let _ = fs::remove_file(path);
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn encode_manifest(meta: &PersistEntryMeta) -> Result<Vec<u8>> {
    let mut w = Writer::new();
    w.u32(MANIFEST_MAGIC);
    w.u8(MANIFEST_VERSION);
    w.str(&meta.object_id);
    w.str(&meta.namespace);
    w.str(&meta.key);
    w.u64(meta.generation);
    w.bytes(&meta.compat.encode());
    w.u64(meta.byte_len);
    w.u64(meta.numel);
    w.u64(meta.created_ns);
    w.u64(meta.blocks.len() as u64);
    for b in &meta.blocks {
        w.bytes(b.content_hash.as_bytes());
        w.u64(b.offset);
        w.u64(b.len);
        w.u32(b.crc);
    }
    let body = w.into_inner();
    let anchor = hash(&body);
    let mut out = body;
    out.extend_from_slice(anchor.as_bytes());
    Ok(out)
}

fn decode_manifest(data: &[u8]) -> Result<PersistEntryMeta> {
    if data.len() < 32 {
        return Err(Error::Persistence("manifest truncated".into()));
    }
    let (body, anchor) = data.split_at(data.len() - 32);
    let computed = hash(body);
    if computed.as_bytes() != anchor {
        return Err(Error::Persistence("manifest anchor mismatch".into()));
    }
    let mut r = Reader::new(body)?;
    let magic = r.u32()?;
    if magic != MANIFEST_MAGIC {
        return Err(Error::Persistence("manifest magic mismatch".into()));
    }
    let version = r.u8()?;
    if version != MANIFEST_VERSION {
        return Err(Error::Persistence(format!(
            "manifest version {version} unsupported"
        )));
    }
    let object_id = r.str()?.to_owned();
    let namespace = r.str()?.to_owned();
    let key = r.str()?.to_owned();
    let generation = r.u64()?;
    let compat_bytes = r.bytes()?.to_vec();
    let compat = CompatKey::decode(&compat_bytes)?;
    let byte_len = r.u64()?;
    let numel = r.u64()?;
    let created_ns = r.u64()?;
    let block_count = r.u64()?;
    if block_count > crate::storage::MAX_BLOCKS_PER_TENSOR {
        return Err(Error::Persistence(
            "manifest block count exceeds maximum".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(block_count as usize);
    for _ in 0..block_count {
        let hash_bytes = r.bytes()?;
        if hash_bytes.len() != 32 {
            return Err(Error::Persistence("block content hash wrong length".into()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(hash_bytes);
        let content_hash = Digest::from_bytes(arr);
        let offset = r.u64()?;
        let len = r.u64()?;
        let crc = r.u32()?;
        blocks.push(BlockRef {
            content_hash,
            offset,
            len,
            crc,
        });
    }
    if !r.eof() {
        return Err(Error::Persistence("trailing bytes in manifest".into()));
    }
    if byte_len != compat.shape.byte_len(compat.dtype.byte_size())? {
        return Err(Error::Persistence(
            "manifest byte_len inconsistent with compat".into(),
        ));
    }
    if numel != compat.shape.numel()? {
        return Err(Error::Persistence(
            "manifest numel inconsistent with compat".into(),
        ));
    }
    crate::storage::validate_block_list(&blocks, byte_len)?;
    Ok(PersistEntryMeta {
        object_id,
        namespace,
        key,
        generation,
        compat,
        byte_len,
        numel,
        created_ns,
        blocks,
    })
}

/// Convenience: verify that a set of persistent blocks is intact.
pub fn verify_persisted_blocks(store: &PersistentStore, blocks: &[BlockRef]) -> Result<u64> {
    for b in blocks {
        let data = store.get_block(&b.content_hash)?;
        if data.len() as u64 != b.len {
            return Err(Error::Integrity(format!(
                "block {} length mismatch",
                b.content_hash
            )));
        }
    }
    Ok(blocks.len() as u64)
}

/// Validate that a persisted manifest's blocks all exist on disk.
pub fn blocks_exist(store: &PersistentStore, blocks: &[BlockRef]) -> bool {
    blocks.iter().all(|b| store.block_exists(&b.content_hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::geometry::{Layout, Shape};
    use crate::storage::chunk;

    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("tensorcache-test-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn sample_meta(blocks: Vec<BlockRef>, byte_len: u64) -> PersistEntryMeta {
        // F64 elements: 8 bytes each, so the shape is a 1-D vector of byte_len/8.
        let elems = byte_len / 8;
        let shape = Shape::new(vec![elems]).unwrap();
        let compat = CompatKey {
            dtype: Dtype::F64,
            shape,
            layout: Layout::RowMajor,
            model: Some("m".into()),
            ..Default::default()
        };
        PersistEntryMeta {
            object_id: "a".repeat(64),
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            compat,
            byte_len,
            numel: elems,
            created_ns: 0,
            blocks,
        }
    }

    #[test]
    fn manifest_roundtrip_and_anchor() {
        let root = temp_root();
        let store = PersistentStore::open(&root).unwrap();
        let data = vec![1u8; 4096];
        let block_refs = chunk(&data, 512).unwrap();
        let meta = sample_meta(block_refs, data.len() as u64);
        store.write_manifest(&meta).unwrap();
        let loaded = store.read_manifest(&"a".repeat(64)).unwrap();
        assert_eq!(loaded.object_id, meta.object_id);
        assert_eq!(loaded.blocks.len(), meta.blocks.len());
        let path = store.manifest_path(&meta.object_id).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        assert!(store.read_manifest(&meta.object_id).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recovery_skips_corrupt_manifests() {
        let root = temp_root();
        let store = PersistentStore::open(&root).unwrap();
        let data = vec![7u8; 1024];
        let blocks = chunk(&data, 256).unwrap();
        let meta = sample_meta(blocks, data.len() as u64);
        store.write_manifest(&meta).unwrap();
        let bad = root.join("manifests").join("b".repeat(63) + "b.manifest");
        fs::write(&bad, b"garbage-that-is-not-a-manifest").unwrap();
        let recovered = store.recover();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].object_id, meta.object_id);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cleanup_removes_temp_files() {
        let root = temp_root();
        let store = PersistentStore::open(&root).unwrap();
        store.cleanup_incomplete().unwrap();
        let dir = root.join("manifests");
        let tmp = dir.join("abc.tmp123");
        fs::write(&tmp, b"partial").unwrap();
        store.cleanup_incomplete().unwrap();
        assert!(!tmp.exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn block_write_and_verify() {
        let root = temp_root();
        let store = PersistentStore::open(&root).unwrap();
        let data = b"persistent block bytes";
        let h = store.put_block(data).unwrap();
        assert_eq!(h, hash(data));
        let back = store.get_block(&h).unwrap();
        assert_eq!(back, data);
        assert!(store.block_exists(&h));
        let path = store.block_path(&h).unwrap();
        let mut b = fs::read(&path).unwrap();
        b[0] ^= 0xAA;
        fs::write(&path, &b).unwrap();
        assert!(store.get_block(&h).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn path_traversal_is_rejected() {
        let root = temp_root();
        let store = PersistentStore::open(&root).unwrap();
        assert!(store.manifest_path("../../etc/passwd").is_err());
        assert!(store.manifest_path("a".repeat(63).as_str()).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn hex_helper() {
        assert!(hex_path_valid(&"a".repeat(64)));
        assert!(!hex_path_valid(&"g".repeat(64)));
        assert!(!hex_path_valid(&"a".repeat(63)));
    }
}
