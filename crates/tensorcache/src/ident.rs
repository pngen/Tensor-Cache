#![forbid(unsafe_code)]
//! Tensor object identity.
//!
//! Every reusable tensor entry has stable identity independent of physical
//! location. An entry is addressed by a namespace, a semantic cache key and a
//! generation (version); together these form the address, and the stable
//! object identity is the SHA-256 of a canonical encoding of that address.
//!
//! Generation is part of identity so that a stale version can never be
//! silently served for a request for a newer one. Address and compat identity
//! are orthogonal: the address says WHAT logical state this is; the compat
//! key says whether it is SAFE to reuse structurally.

use crate::error::{Error, Result};
use crate::hash::{hash, parse_hex, Digest};
use crate::wire::Writer;

/// A logical tensor address: namespace + semantic cache key + generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Address {
    pub namespace: String,
    pub key: String,
    pub generation: u64,
}

impl Address {
    pub fn new(namespace: impl Into<String>, key: impl Into<String>, generation: u64) -> Self {
        Address {
            namespace: namespace.into(),
            key: key.into(),
            generation,
        }
    }

    /// Encode the address into its canonical identity bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.u8(1); // address version
        w.str(&self.namespace);
        w.str(&self.key);
        w.u64(self.generation);
        w.into_inner()
    }

    /// The stable object identity: SHA-256 over the canonical address bytes.
    pub fn object_id(&self) -> ObjectId {
        ObjectId::from_digest(hash(&self.encode()))
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} / {} @ gen{}",
            self.namespace, self.key, self.generation
        )
    }
}

/// A stable 256-bit object identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(Digest);

impl ObjectId {
    pub fn from_digest(d: Digest) -> Self {
        ObjectId(d)
    }

    pub fn digest(&self) -> &Digest {
        &self.0
    }

    /// The 64-character lower-case hex form used as a storage key / file name.
    pub fn to_hex(&self) -> String {
        self.0.to_hex()
    }

    /// Parse an object id from its canonical hex form.
    pub fn from_hex(s: &str) -> Result<Self> {
        let bytes = parse_hex(s)?;
        if bytes.len() != 32 {
            return Err(Error::InvalidArgument("object id must be 32 bytes".into()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(ObjectId(Digest::from_bytes(arr)))
    }
}

impl std::fmt::Display for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_id_is_stable_and_hex_roundtrips() {
        let a = Address::new("ns", "k1", 3);
        let oid = a.object_id();
        assert_eq!(oid, a.object_id());
        assert_eq!(ObjectId::from_hex(&oid.to_hex()).unwrap(), oid);
        assert_eq!(oid.to_hex().len(), 64);
    }

    #[test]
    fn object_id_distinguishes_generation() {
        let a1 = Address::new("ns", "k", 1).object_id();
        let a2 = Address::new("ns", "k", 2).object_id();
        assert_ne!(a1, a2);
    }

    #[test]
    fn object_id_distinguishes_namespace_and_key() {
        assert_ne!(
            Address::new("a", "k", 1).object_id(),
            Address::new("b", "k", 1).object_id()
        );
        assert_ne!(
            Address::new("a", "k", 1).object_id(),
            Address::new("a", "k2", 1).object_id()
        );
    }
}
