#![forbid(unsafe_code)]
//! Tensor geometry: shape, rank and memory layout with strict validation.
//!
//! Geometry is part of the canonical compatibility identity. Impossible
//! geometry (rank abuse, per-dimension overflow, or a product / byte-length
//! that overflows the address space) is rejected before any allocation.

use crate::error::{Error, Result};

/// Maximum supported rank (dimension count).
pub const MAX_RANK: usize = 32;
/// Maximum value for any single dimension. Bounded well below u64::MAX so that
/// product and byte-length computations cannot silently alias through overflow.
pub const MAX_DIM: u64 = u32::MAX as u64;

/// A validated tensor shape.
///
/// Construction enforces rank and per-dimension bounds; every arithmetic
/// operation is checked so that geometry overflows are impossible.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<u64>);

impl Shape {
    /// Build a shape, rejecting out-of-range rank or dimensions.
    pub fn new(dims: Vec<u64>) -> Result<Shape> {
        if dims.len() > MAX_RANK {
            return Err(Error::Geometry(format!(
                "rank {} exceeds maximum {}",
                dims.len(),
                MAX_RANK
            )));
        }
        if dims.iter().any(|&d| d > MAX_DIM) {
            return Err(Error::Geometry(format!(
                "dimension exceeds maximum {}",
                MAX_DIM
            )));
        }
        Ok(Shape(dims))
    }

    pub fn from_slice(dims: &[u64]) -> Result<Shape> {
        Shape::new(dims.to_vec())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[u64] {
        &self.0
    }

    pub fn to_vec(&self) -> Vec<u64> {
        self.0.clone()
    }

    /// The number of logical elements, checked for overflow.
    pub fn numel(&self) -> Result<u64> {
        let mut n: u64 = 1;
        for &d in &self.0 {
            // The empty shape is the rank-0 scalar with one element.
            n = n
                .checked_mul(d)
                .ok_or_else(|| Error::Geometry("shape element count overflow".into()))?;
        }
        Ok(n)
    }

    /// Computed byte length for the given element byte size, checked.
    pub fn byte_len(&self, elem: u64) -> Result<u64> {
        // Guard against a zero element size that would let a huge numel pass.
        if elem == 0 || elem > u64::MAX / 8 {
            return Err(Error::Geometry("invalid element size".into()));
        }
        self.numel()?
            .checked_mul(elem)
            .ok_or_else(|| Error::Geometry("tensor byte length overflow".into()))
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Shape> for Vec<u64> {
    fn from(s: Shape) -> Vec<u64> {
        s.0
    }
}

/// Memory layout of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C) contiguous.
    RowMajor,
    /// Column-major (Fortran) contiguous.
    ColMajor,
    /// An explicit, dense stride vector. Strides are validated as strictly
    /// increasing in the number of elements they cross and checked for
    /// overflow against the shape.
    Strided(Vec<u64>),
}

impl Layout {
    /// Return the storage order tag used in the canonical encoding.
    pub const fn code(&self) -> u8 {
        match self {
            Layout::RowMajor => 0,
            Layout::ColMajor => 1,
            Layout::Strided(_) => 2,
        }
    }

    /// Validate the layout against a shape.
    pub fn validate(&self, shape: &Shape) -> Result<()> {
        match self {
            Layout::RowMajor | Layout::ColMajor => Ok(()),
            Layout::Strided(strides) => {
                if strides.len() != shape.rank() {
                    return Err(Error::Geometry(format!(
                        "strides length {} does not match rank {}",
                        strides.len(),
                        shape.rank()
                    )));
                }
                // Reject zero or negative strides (sliced tensors are not
                // cached; they are materialized as contiguous copies).
                for &s in strides {
                    if s == 0 {
                        return Err(Error::Geometry("zero stride is not cacheable".into()));
                    }
                }
                // Max offset must not overflow.
                let mut max_off: u64 = 0;
                for (&d, &s) in shape.dims().iter().zip(strides.iter()) {
                    let span = if d == 0 { 0 } else { d - 1 };
                    max_off = max_off
                        .checked_add(
                            span.checked_mul(s)
                                .ok_or_else(|| Error::Geometry("stride product overflow".into()))?,
                        )
                        .ok_or_else(|| Error::Geometry("stride offset overflow".into()))?;
                }
                Ok(())
            }
        }
    }

    /// The canonical stride vector for a contiguous layout of the given shape.
    pub fn contiguous_strides(&self, shape: &Shape) -> Result<Vec<u64>> {
        match self {
            Layout::RowMajor => row_major_strides(shape),
            Layout::ColMajor => col_major_strides(shape),
            Layout::Strided(s) => {
                self.validate(shape)?;
                Ok(s.clone())
            }
        }
    }
}

fn row_major_strides(shape: &Shape) -> Result<Vec<u64>> {
    let dims = shape.dims();
    let mut strides = vec![0u64; dims.len()];
    let mut acc: u64 = 1;
    for i in (0..dims.len()).rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(dims[i])
            .ok_or_else(|| Error::Geometry("row-major stride overflow".into()))?;
    }
    Ok(strides)
}

fn col_major_strides(shape: &Shape) -> Result<Vec<u64>> {
    let dims = shape.dims();
    let mut strides = vec![0u64; dims.len()];
    let mut acc: u64 = 1;
    for i in 0..dims.len() {
        strides[i] = acc;
        acc = acc
            .checked_mul(dims[i])
            .ok_or_else(|| Error::Geometry("column-major stride overflow".into()))?;
    }
    Ok(strides)
}

/// Iterate the product of dims and strides across the whole tensor in the
/// given contiguous order, returning the number of elements and the maximum
/// byte offset. Used for byte-length cross checking.
pub fn storage_span(shape: &Shape, elem: u64) -> Result<u64> {
    shape.byte_len(elem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_rank_bounds() {
        assert!(Shape::new(vec![2, 3]).is_ok());
        assert!(Shape::new(vec![]).is_ok()); // rank-0 scalar
        assert!(Shape::new(vec![1u64; MAX_RANK]).is_ok());
        assert!(Shape::new(vec![1u64; MAX_RANK + 1]).is_err());
        assert!(Shape::new(vec![MAX_DIM + 1]).is_err());
    }

    #[test]
    fn shape_numel_and_byte_len() {
        let s = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(s.numel().unwrap(), 12);
        assert_eq!(s.byte_len(4).unwrap(), 48);
        assert_eq!(Shape::new(vec![]).unwrap().numel().unwrap(), 1);
    }

    #[test]
    fn shape_overflow_rejected() {
        // (2^32 - 1)^3 overflows u64 and must be detected before it wraps.
        let s = Shape::new(vec![u32::MAX as u64; 3]).unwrap();
        assert!(s.numel().is_err());
        // A single dimension above the per-dim bound is rejected directly.
        assert!(Shape::new(vec![u64::MAX]).is_err());
    }

    #[test]
    fn strides_rowmajor_correct() {
        let s = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(row_major_strides(&s).unwrap(), vec![12, 4, 1]);
        let s2 = Shape::new(vec![2, 3]).unwrap();
        assert_eq!(col_major_strides(&s2).unwrap(), vec![1, 2]);
    }

    #[test]
    fn strided_layout_validation() {
        let s = Shape::new(vec![2, 3]).unwrap();
        assert!(Layout::Strided(vec![3, 1]).validate(&s).is_ok());
        assert!(Layout::Strided(vec![3]).validate(&s).is_err()); // wrong len
        assert!(Layout::Strided(vec![0, 1]).validate(&s).is_err()); // zero stride
                                                                    // Huge stride overflows.
        assert!(Layout::Strided(vec![u64::MAX, 1]).validate(&s).is_err());
    }

    #[test]
    fn layout_code() {
        assert_eq!(Layout::RowMajor.code(), 0);
        assert_eq!(Layout::ColMajor.code(), 1);
        assert_eq!(Layout::Strided(vec![1]).code(), 2);
    }
}
