#![forbid(unsafe_code)]
//! Canonical tensor compatibility identity.
//!
//! Tensor Cache must never reuse a tensor across incompatible geometry, dtype,
//! layout, model, runtime, revision, quantization or semantic meaning. A false
//! reuse hit is a correctness failure. To make reuse decisions safe and
//! deterministic, every reusable tensor has a *compatibility identity*: a
//! cheap, structural fingerprint computed by hashing a canonical, versioned,
//! unambiguous encoding of the exact fields that gate reuse.
//!
//! The canonical encoding is collision-safe by construction: it uses a leading
//! version byte and length-prefixed strings/byte blobs, so no delimiter
//! trick can make two distinct tensors encode identically.

use crate::dtype::{Dtype, Endianness, Mutability, QuantKind};
use crate::error::{Error, Result};
use crate::geometry::{Layout, Shape};
use crate::hash::{hash, Digest};
use crate::wire::{Reader, Writer};

/// The version of the canonical compatibility encoding.
pub const CANON_VERSION: u8 = 1;

/// The full set of fields that gate whether one tensor may safely be reused
/// as another. A difference in ANY of these fields changes the compatibility
/// identity and therefore forbids reuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatKey {
    pub model: Option<String>,
    pub model_revision: Option<String>,
    pub runtime_version: Option<String>,
    pub operation: Option<String>,
    pub dtype: Dtype,
    pub shape: Shape,
    pub layout: Layout,
    pub endianness: Endianness,
    pub device: Option<String>,
    pub precision: Option<String>,
    pub quant: QuantKind,
    pub mutability: Mutability,
}

impl Default for CompatKey {
    fn default() -> Self {
        CompatKey {
            model: None,
            model_revision: None,
            runtime_version: None,
            operation: None,
            dtype: Dtype::F32,
            shape: Shape::new(vec![]).expect("empty shape is valid"),
            layout: Layout::RowMajor,
            endianness: Endianness::Little,
            device: None,
            precision: None,
            quant: QuantKind::None,
            mutability: Mutability::Immutable,
        }
    }
}

impl CompatKey {
    /// Encode the compatibility key into its canonical, versioned byte form.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.u8(CANON_VERSION);
        write_opt_str(&mut w, &self.model);
        write_opt_str(&mut w, &self.model_revision);
        write_opt_str(&mut w, &self.runtime_version);
        write_opt_str(&mut w, &self.operation);
        w.u8(self.dtype.code());
        w.u64(self.shape.rank() as u64);
        for &d in self.shape.dims() {
            w.u64(d);
        }
        w.u8(self.layout.code());
        if let Layout::Strided(strides) = &self.layout {
            w.u64(strides.len() as u64);
            for &s in strides {
                w.u64(s);
            }
        }
        w.u8(self.endianness.code());
        write_opt_str(&mut w, &self.device);
        write_opt_str(&mut w, &self.precision);
        w.u8(self.quant.code());
        w.u8(self.mutability.code());
        w.into_inner()
    }

    /// Decode a compatibility key from its canonical byte form, rejecting a
    /// version mismatch or any malformed field.
    pub fn decode(data: &[u8]) -> Result<CompatKey> {
        let mut r = Reader::new(data)?;
        let version = r.u8()?;
        if version != CANON_VERSION {
            return Err(Error::Protocol(format!(
                "unsupported compat encoding version {version}"
            )));
        }
        let model = read_opt_str(&mut r)?;
        let model_revision = read_opt_str(&mut r)?;
        let runtime_version = read_opt_str(&mut r)?;
        let operation = read_opt_str(&mut r)?;
        let dtype = Dtype::from_code(r.u8()?)?;
        let rank = r.u64()?;
        if rank > crate::geometry::MAX_RANK as u64 {
            return Err(Error::Protocol(format!("compat rank {rank} exceeds max")));
        }
        let mut dims = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dims.push(r.u64()?);
        }
        // Each dim is bounded during Shape::new; but guard an absurd value
        // before allocation growth.
        for &d in &dims {
            if d > crate::geometry::MAX_DIM {
                return Err(Error::Protocol("compat dimension exceeds max".into()));
            }
        }
        let shape = Shape::new(dims)?;
        let layout_code = r.u8()?;
        let layout = match layout_code {
            0 => Layout::RowMajor,
            1 => Layout::ColMajor,
            2 => {
                let n = r.u64()?;
                if n > crate::geometry::MAX_RANK as u64 {
                    return Err(Error::Protocol("compat strides exceed max".into()));
                }
                let mut strides = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    strides.push(r.u64()?);
                }
                Layout::Strided(strides)
            }
            other => return Err(Error::Protocol(format!("unknown layout tag {other}"))),
        };
        layout.validate(&shape)?;
        let endianness = Endianness::from_code(r.u8()?)?;
        let device = read_opt_str(&mut r)?;
        let precision = read_opt_str(&mut r)?;
        let quant = QuantKind::from_code(r.u8()?)?;
        let mutability = Mutability::from_code(r.u8()?)?;
        if !r.eof() {
            return Err(Error::Protocol("trailing bytes in compat encoding".into()));
        }
        Ok(CompatKey {
            model,
            model_revision,
            runtime_version,
            operation,
            dtype,
            shape,
            layout,
            endianness,
            device,
            precision,
            quant,
            mutability,
        })
    }

    /// The compatibility identity: the SHA-256 digest of the canonical bytes.
    pub fn compat_id(&self) -> Digest {
        hash(&self.encode())
    }

    /// Whether two keys have identical compatibility identity.
    pub fn same_compat(&self, other: &CompatKey) -> bool {
        self.compat_id() == other.compat_id()
    }

    /// A concise, human-readable rendering of the key for diagnostics.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!("dtype={}", self.dtype));
        parts.push(format!("shape={:?}", self.shape.dims()));
        parts.push(format!("layout={:?}", self.layout));
        if let Some(m) = &self.model {
            parts.push(format!("model={m}"));
        }
        if let Some(r) = &self.model_revision {
            parts.push(format!("rev={r}"));
        }
        if let Some(r) = &self.runtime_version {
            parts.push(format!("runtime={r}"));
        }
        if let Some(o) = &self.operation {
            parts.push(format!("op={o}"));
        }
        if let Some(d) = &self.device {
            parts.push(format!("device={d}"));
        }
        if let Some(p) = &self.precision {
            parts.push(format!("precision={p}"));
        }
        if self.quant != QuantKind::None {
            parts.push(format!("quant={:?}", self.quant));
        }
        parts.push(format!("mut={:?}", self.mutability));
        parts.join(" ")
    }
}

fn write_opt_str(w: &mut Writer, v: &Option<String>) {
    match v {
        Some(s) => {
            w.bool(true);
            w.str(s);
        }
        None => {
            w.bool(false);
        }
    }
}

fn read_opt_str(r: &mut Reader<'_>) -> Result<Option<String>> {
    let present = r.bool()?;
    if present {
        Ok(Some(r.str()?.to_owned()))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_key() -> CompatKey {
        CompatKey {
            dtype: Dtype::F32,
            shape: Shape::new(vec![2, 3]).unwrap(),
            layout: Layout::RowMajor,
            model: Some("m1".into()),
            ..Default::default()
        }
    }

    #[test]
    fn same_key_same_id() {
        assert_eq!(base_key().compat_id(), base_key().compat_id());
    }

    #[test]
    fn dtype_difference_rejected() {
        let mut a = base_key();
        let b = base_key();
        a.dtype = Dtype::F16;
        assert_ne!(a.compat_id(), b.compat_id());
    }

    #[test]
    fn shape_difference_rejected() {
        let mut a = base_key();
        let b = base_key();
        a.shape = Shape::new(vec![3, 2]).unwrap();
        assert_ne!(a.compat_id(), b.compat_id());
    }

    #[test]
    fn layout_difference_rejected() {
        let mut a = base_key();
        let b = base_key();
        a.layout = Layout::ColMajor;
        assert_ne!(a.compat_id(), b.compat_id());
    }

    #[test]
    fn model_and_revision_difference_rejected() {
        let b = base_key();
        let mut a = base_key();
        a.model = Some("m2".into());
        assert_ne!(a.compat_id(), b.compat_id());
        let mut c = base_key();
        c.model_revision = Some("r2".into());
        assert_ne!(c.compat_id(), b.compat_id());
    }

    #[test]
    fn operation_semantic_difference_rejected() {
        let b = base_key();
        let mut a = base_key();
        a.operation = Some("layer2".into());
        assert_ne!(a.compat_id(), b.compat_id());
    }

    #[test]
    fn encode_decode_roundtrip() {
        let k = CompatKey {
            model: Some("model-a".into()),
            model_revision: Some("v7".into()),
            runtime_version: Some("1.2.3".into()),
            operation: Some("embed".into()),
            dtype: Dtype::F16,
            shape: Shape::new(vec![128, 4096]).unwrap(),
            layout: Layout::Strided(vec![4096, 1]),
            endianness: Endianness::Big,
            device: Some("cuda".into()),
            precision: Some("bf16".into()),
            quant: QuantKind::Nf4,
            mutability: Mutability::Immutable,
        };
        let enc = k.encode();
        let dec = CompatKey::decode(&enc).unwrap();
        assert_eq!(dec.compat_id(), k.compat_id());
        assert_eq!(dec, k);
    }

    #[test]
    fn option_none_vs_some_is_unambiguous() {
        let a = base_key();
        let mut b = base_key();
        b.model = Some("".into());
        assert_ne!(a.compat_id(), b.compat_id());
    }

    #[test]
    fn bad_version_rejected() {
        let enc = base_key().encode();
        let mut bad = enc.clone();
        bad[0] = 99;
        assert!(CompatKey::decode(&bad).is_err());
    }

    #[test]
    fn string_escaping_is_not_an_issue() {
        // Strings containing separator-like characters must not collide.
        let a = CompatKey {
            operation: Some("a|b".into()),
            ..base_key()
        };
        let b = CompatKey {
            operation: Some("a".into()),
            ..base_key()
        };
        let c = CompatKey {
            operation: Some("b".into()),
            ..base_key()
        };
        assert_ne!(a.compat_id(), b.compat_id());
        assert_ne!(b.compat_id(), c.compat_id());
    }
}
