#![forbid(unsafe_code)]
//! Primitive tensor element types and small compat dimensions.
//!
//! The dtype is part of the canonical compatibility identity: two tensors are
//! only safely reusable when their element types are identical. The unit of
//! element size is bytes for ordinary types.

use crate::error::{Error, Result};

/// A primitive tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    F64 = 1,
    F32 = 2,
    F16 = 3,
    BF16 = 4,
    F8 = 5,
    I64 = 6,
    I32 = 7,
    I16 = 8,
    I8 = 9,
    U64 = 10,
    U32 = 11,
    U16 = 12,
    U8 = 13,
    Bool = 14,
}

impl Dtype {
    /// A stable one-byte tag used in serialization.
    pub const fn code(&self) -> u8 {
        *self as u8
    }

    /// Parse a dtype from its wire tag.
    pub fn from_code(c: u8) -> Result<Self> {
        Ok(match c {
            1 => Dtype::F64,
            2 => Dtype::F32,
            3 => Dtype::F16,
            4 => Dtype::BF16,
            5 => Dtype::F8,
            6 => Dtype::I64,
            7 => Dtype::I32,
            8 => Dtype::I16,
            9 => Dtype::I8,
            10 => Dtype::U64,
            11 => Dtype::U32,
            12 => Dtype::U16,
            13 => Dtype::U8,
            14 => Dtype::Bool,
            _ => return Err(Error::InvalidArgument(format!("unknown dtype code {c}"))),
        })
    }

    /// Byte size of one element.
    pub const fn byte_size(&self) -> u64 {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F8 | Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }

    /// Whether the type is a floating-point type.
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F8
        )
    }

    /// The human-readable type name.
    pub const fn name(&self) -> &'static str {
        match self {
            Dtype::F64 => "f64",
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::F8 => "f8",
            Dtype::I64 => "i64",
            Dtype::I32 => "i32",
            Dtype::I16 => "i16",
            Dtype::I8 => "i8",
            Dtype::U64 => "u64",
            Dtype::U32 => "u32",
            Dtype::U16 => "u16",
            Dtype::U8 => "u8",
            Dtype::Bool => "bool",
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Byte order for multi-byte element payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Endianness {
    Little = 0,
    Big = 1,
}

impl Endianness {
    pub const fn code(&self) -> u8 {
        *self as u8
    }
    pub fn from_code(c: u8) -> Result<Self> {
        Ok(match c {
            0 => Endianness::Little,
            1 => Endianness::Big,
            _ => return Err(Error::InvalidArgument(format!("unknown endianness {c}"))),
        })
    }
}

/// Whether an entry may be mutated in place after admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Mutability {
    Immutable = 0,
    Mutable = 1,
}

impl Mutability {
    pub const fn code(&self) -> u8 {
        *self as u8
    }
    pub fn from_code(c: u8) -> Result<Self> {
        Ok(match c {
            0 => Mutability::Immutable,
            1 => Mutability::Mutable,
            _ => return Err(Error::InvalidArgument(format!("unknown mutability {c}"))),
        })
    }
}

/// A quantization scheme identity, used to forbid reuse across incompatible
/// quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuantKind {
    None = 0,
    Int4 = 1,
    Int8 = 2,
    Nf4 = 3,
}

impl QuantKind {
    pub const fn code(&self) -> u8 {
        *self as u8
    }
    pub fn from_code(c: u8) -> Result<Self> {
        Ok(match c {
            0 => QuantKind::None,
            1 => QuantKind::Int4,
            2 => QuantKind::Int8,
            3 => QuantKind::Nf4,
            _ => return Err(Error::InvalidArgument(format!("unknown quant kind {c}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_roundtrip() {
        for d in [
            Dtype::F64,
            Dtype::F32,
            Dtype::F16,
            Dtype::BF16,
            Dtype::F8,
            Dtype::I64,
            Dtype::I32,
            Dtype::I16,
            Dtype::I8,
            Dtype::U64,
            Dtype::U32,
            Dtype::U16,
            Dtype::U8,
            Dtype::Bool,
        ] {
            assert_eq!(Dtype::from_code(d.code()).unwrap(), d);
        }
        assert!(Dtype::from_code(200).is_err());
    }

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(Dtype::F32.byte_size(), 4);
        assert_eq!(Dtype::F16.byte_size(), 2);
        assert_eq!(Dtype::Bool.byte_size(), 1);
        assert_eq!(Dtype::U64.byte_size(), 8);
    }

    #[test]
    fn endianness_and_mutability_roundtrip() {
        assert_eq!(
            Endianness::from_code(Endianness::Big.code()).unwrap(),
            Endianness::Big
        );
        assert_eq!(
            Mutability::from_code(Mutability::Mutable.code()).unwrap(),
            Mutability::Mutable
        );
    }
}
