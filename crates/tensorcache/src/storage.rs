#![forbid(unsafe_code)]
//! Tensor storage model: logical tensors, physical blocks, content addressing
//! and reconstruction.
//!
//! A tensor payload is a contiguous byte buffer. It is split into blocks of a
//! configurable size. Each block carries a content hash (SHA-256) used for
//! deduplication, a CRC-32C checksum used for integrity, a byte offset within
//! the tensor and a length. Blocks are content-addressed, so byte-identical
//! blocks shared by two logical objects occupy one physical copy. A tensor is
//! reconstructed by reassembling its ordered blocks.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::crc::crc32c;
use crate::error::{Error, Result};
use crate::hash::{hash, Digest};

/// Default block size (1 MiB).
pub const DEFAULT_BLOCK_SIZE: u64 = 1 << 20;
/// Hard upper bound on the number of blocks for a single tensor, to prevent
/// block-count explosions from a malformed manifest.
pub const MAX_BLOCKS_PER_TENSOR: u64 = 1 << 24;

/// A reference to one physical block of a logical tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRef {
    /// Content address (SHA-256 of the raw block bytes).
    pub content_hash: Digest,
    /// Byte offset of this block within the logical tensor.
    pub offset: u64,
    /// Number of bytes in this block.
    pub len: u64,
    /// CRC-32C of the raw block bytes.
    pub crc: u32,
}

/// Split a contiguous payload into ordered blocks.
pub fn chunk(bytes: &[u8], block_size: u64) -> Result<Vec<BlockRef>> {
    if block_size == 0 {
        return Err(Error::InvalidArgument("block size must be nonzero".into()));
    }
    let total = bytes.len() as u64;
    let n_blocks = total.div_ceil(block_size);
    if n_blocks > MAX_BLOCKS_PER_TENSOR {
        return Err(Error::Geometry("block count exceeds maximum".into()));
    }
    let mut out = Vec::new();
    let mut offset: u64 = 0;
    while offset < total {
        let end = offset + block_size.min(total - offset);
        let block = &bytes[offset as usize..end as usize];
        let content_hash = hash(block);
        let crc = crc32c(block);
        out.push(BlockRef {
            content_hash,
            offset,
            len: block.len() as u64,
            crc,
        });
        offset = end;
    }
    Ok(out)
}

/// Validate that a set of ordered block references exactly covers a byte range
/// of the given length with no gaps, overlaps or out-of-range offsets, and
/// that there are not too many blocks.
pub fn validate_block_list(blocks: &[BlockRef], byte_len: u64) -> Result<()> {
    if blocks.len() as u64 > MAX_BLOCKS_PER_TENSOR {
        return Err(Error::Geometry("block count exceeds maximum".into()));
    }
    let mut expected: u64 = 0;
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    for b in blocks {
        if b.len == 0 {
            return Err(Error::Geometry("zero-length block".into()));
        }
        if b.offset != expected {
            return Err(Error::Geometry(format!(
                "block gap/overlap at offset {} (expected {})",
                b.offset, expected
            )));
        }
        let end = b.offset + b.len;
        if end > byte_len {
            return Err(Error::Geometry("block extends past tensor length".into()));
        }
        if !seen.insert((b.offset, b.len)) {
            return Err(Error::Geometry("duplicate block identity".into()));
        }
        expected = end;
    }
    if expected != byte_len {
        return Err(Error::Geometry(format!(
            "blocks cover {} bytes but tensor is {byte_len} bytes",
            expected
        )));
    }
    Ok(())
}

/// A content-addressed block arena that deduplicates identical block bytes and
/// tracks reference counts. A block is freed only when its last referring
/// tensor is released, so two logical objects may safely share physical bytes
/// without their identities collapsing.
#[derive(Default)]
pub struct BlockArena {
    blocks: HashMap<Digest, Arc<[u8]>>,
    refs: HashMap<Digest, u64>,
    bytes: u64,
}

impl BlockArena {
    pub fn new() -> Self {
        BlockArena::default()
    }

    /// The total number of unique block bytes held.
    pub fn bytes_used(&self) -> u64 {
        self.bytes
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Register a block and acquire one reference to it.
    pub fn acquire(&mut self, bytes: &[u8]) -> BlockRef {
        let h = hash(bytes);
        let len = bytes.len() as u64;
        let crc = crc32c(bytes);
        match self.blocks.get(&h) {
            Some(_) => {
                *self.refs.get_mut(&h).unwrap() += 1;
            }
            None => {
                self.blocks.insert(h, Arc::from(bytes));
                self.refs.insert(h, 1);
                self.bytes += len;
            }
        }
        BlockRef {
            content_hash: h,
            offset: 0,
            len,
            crc,
        }
    }

    /// Register a block with an explicit offset (for reconstructing exactly one
    /// tensor). Deduplication is still by content hash.
    pub fn acquire_at(&mut self, bytes: &[u8], offset: u64) -> BlockRef {
        let mut r = self.acquire(bytes);
        r.offset = offset;
        r
    }

    /// Release one reference to a block; free bytes when it reaches zero.
    /// Returns true if the block's physical bytes were freed (last reference).
    pub fn release(&mut self, content_hash: &Digest) -> bool {
        if let Some(r) = self.refs.get_mut(content_hash) {
            *r -= 1;
            if *r == 0 {
                if let Some(block) = self.blocks.remove(content_hash) {
                    self.bytes -= block.len() as u64;
                }
                self.refs.remove(content_hash);
                return true;
            }
        }
        false
    }

    /// Fetch a reference-counted copy of a block bytes (does not change refcount).
    pub fn get(&self, content_hash: &Digest) -> Result<Arc<[u8]>> {
        self.blocks
            .get(content_hash)
            .cloned()
            .ok_or_else(|| Error::Reconstruct(format!("block {} missing", content_hash)))
    }

    /// Whether a block is present.
    pub fn contains(&self, content_hash: &Digest) -> bool {
        self.blocks.contains_key(content_hash)
    }
}

/// Reassemble a contiguous tensor from ordered block references using a block
/// provider. The block list is validated before any allocation.
pub fn reconstruct<F>(blocks: &[BlockRef], byte_len: u64, get: F) -> Result<Vec<u8>>
where
    F: Fn(&Digest) -> Result<Arc<[u8]>>,
{
    validate_block_list(blocks, byte_len)?;
    let total: usize = usize::try_from(byte_len)
        .map_err(|_| Error::Geometry("tensor length does not fit host usize".into()))?;
    let mut out = vec![0u8; total];
    for b in blocks {
        let data = get(&b.content_hash)?;
        if data.len() as u64 != b.len {
            return Err(Error::Integrity(format!(
                "block {} length {} does not match declared {}",
                b.content_hash,
                data.len(),
                b.len
            )));
        }
        let crc = crc32c(&data);
        if crc != b.crc {
            return Err(Error::Integrity(format!(
                "block {} checksum mismatch",
                b.content_hash
            )));
        }
        let start = b.offset as usize;
        let end = start + b.len as usize;
        out[start..end].copy_from_slice(&data);
    }
    Ok(out)
}

/// Verify the integrity of a reconstructed tensor against its block list.
pub fn verify_reconstructed(
    blocks: &[BlockRef],
    byte_len: u64,
    get: impl Fn(&Digest) -> Result<Arc<[u8]>>,
) -> Result<Vec<u8>> {
    reconstruct(blocks, byte_len, get)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(n: usize) -> Vec<u8> {
        // Prime-period pattern (251) so distinct 512-byte blocks never collide
        // in the dedup arena for this test.
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn chunk_reconstruct_roundtrip() {
        let data = payload(3000);
        let blocks = chunk(&data, 512).unwrap();
        assert_eq!(blocks.len(), 6);
        let mut a2 = BlockArena::new();
        let mut refs = Vec::new();
        for blk in &blocks {
            let start = blk.offset as usize;
            let end = start + blk.len as usize;
            refs.push(a2.acquire_at(&data[start..end], blk.offset));
        }
        let recon = reconstruct(&refs, data.len() as u64, |h| a2.get(h)).unwrap();
        assert_eq!(recon, data);
        assert_eq!(a2.bytes_used(), data.len() as u64);
    }

    #[test]
    fn dedup_shares_physical_bytes() {
        let mut arena = BlockArena::new();
        let block = b"identical-block-payload-1234";
        let a = arena.acquire(block);
        let b = arena.acquire(block);
        assert_eq!(a.content_hash, b.content_hash);
        assert_eq!(arena.bytes_used(), block.len() as u64);
        assert_eq!(arena.block_count(), 1);
        arena.release(&a.content_hash);
        assert!(arena.contains(&a.content_hash));
        arena.release(&b.content_hash);
        assert!(!arena.contains(&a.content_hash));
        assert_eq!(arena.bytes_used(), 0);
    }

    #[test]
    fn block_list_validation_detects_gaps_and_overlap() {
        let blocks = vec![
            BlockRef {
                content_hash: hash(b"a"),
                offset: 0,
                len: 2,
                crc: 0,
            },
            BlockRef {
                content_hash: hash(b"b"),
                offset: 3,
                len: 2,
                crc: 0,
            },
        ];
        assert!(validate_block_list(&blocks, 4).is_err());
        let overlap = vec![
            BlockRef {
                content_hash: hash(b"a"),
                offset: 0,
                len: 3,
                crc: 0,
            },
            BlockRef {
                content_hash: hash(b"b"),
                offset: 1,
                len: 2,
                crc: 0,
            },
        ];
        assert!(validate_block_list(&overlap, 4).is_err());
        let exact = vec![BlockRef {
            content_hash: hash(b"a"),
            offset: 0,
            len: 4,
            crc: 0,
        }];
        assert!(validate_block_list(&exact, 4).is_ok());
        assert!(validate_block_list(&exact, 5).is_err());
    }

    #[test]
    fn corruption_detected_before_copy() {
        let mut arena = BlockArena::new();
        let data = payload(1000);
        let br = arena.acquire(&data);
        let mut bad = br.clone();
        bad.crc ^= 0xFFFF;
        let res = reconstruct(&[bad.clone()], data.len() as u64, |h| arena.get(h));
        assert!(res.is_err());
    }
}
