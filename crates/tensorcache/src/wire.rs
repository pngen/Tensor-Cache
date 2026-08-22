#![forbid(unsafe_code)]
//! A small, bounds-checked binary codec used by the canonical compatibility
//! encoding and by the framed wire protocol.
//!
//! Numeric values are fixed-width little-endian. Strings and byte blobs are
//! length-prefixed with a u64 so that no delimiter-collision attack is
//! possible. The reader never allocates from a peer-controlled length and
//! never reads past the end of its input.

use crate::error::{Error, Result};

/// An append-only binary writer.
#[derive(Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer { buf: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Writer {
            buf: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn i64(&mut self, v: i64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn bool(&mut self, v: bool) -> &mut Self {
        self.buf.push(if v { 1 } else { 0 });
        self
    }

    /// Write a length-prefixed byte blob.
    pub fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.u64(v.len() as u64);
        self.buf.extend_from_slice(v);
        self
    }

    /// Write a length-prefixed UTF-8 string.
    pub fn str(&mut self, v: &str) -> &mut Self {
        self.bytes(v.as_bytes())
    }

    /// Write an explicit enum tag.
    pub fn tag(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }
}

impl WriteExt for Writer {}

/// Convenience trait so that the writer methods can be chained from generic
/// helpers without borrowing side effects.
pub trait WriteExt: Sized {}

/// A bounds-checked binary reader.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Ok(Reader { data, pos: 0 })
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub fn eof(&self) -> bool {
        self.pos == self.data.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(Error::Protocol(format!(
                "truncated frame: need {n} bytes, have {}",
                self.remaining()
            )));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    /// Read a single byte, requiring at least one remaining byte.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    pub fn bool(&mut self) -> Result<bool> {
        let v = self.u8()?;
        match v {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::Protocol(format!("invalid bool byte {v}"))),
        }
    }

    /// Read a length-prefixed byte blob. The length is validated against the
    /// remaining input before any allocation.
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.u64()?;
        let n: usize = usize::try_from(len)
            .map_err(|_| Error::Protocol("length does not fit in host usize".into()))?;
        self.take(n)
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn str(&mut self) -> Result<&'a str> {
        let b = self.bytes()?;
        std::str::from_utf8(b).map_err(|e| Error::Protocol(format!("invalid UTF-8 in frame: {e}")))
    }

    pub fn tag(&mut self) -> Result<u8> {
        self.u8()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_primitives() {
        let mut w = Writer::new();
        w.u8(0x12)
            .u16(0x3456)
            .u32(0x789abcde)
            .u64(0x0fedcba987654321);
        w.bool(true).bool(false);
        w.str("hello");
        w.bytes(&[1, 2, 3, 4]);

        let mut r = Reader::new(w.as_slice()).unwrap();
        assert_eq!(r.u8().unwrap(), 0x12);
        assert_eq!(r.u16().unwrap(), 0x3456);
        assert_eq!(r.u32().unwrap(), 0x789abcde);
        assert_eq!(r.u64().unwrap(), 0x0fedcba987654321);
        assert!(r.bool().unwrap());
        assert!(!r.bool().unwrap());
        assert_eq!(r.str().unwrap(), "hello");
        assert_eq!(r.bytes().unwrap(), &[1, 2, 3, 4]);
        assert!(r.eof());
    }

    #[test]
    fn truncated_read_rejected() {
        let mut w = Writer::new();
        w.u32(5);
        let mut r = Reader::new(&w.as_slice()[..2]).unwrap();
        assert!(r.u32().is_err());
    }

    #[test]
    fn length_prefix_never_overallocates() {
        // A frame claiming a huge length but containing few bytes.
        let mut w = Writer::new();
        w.u64(u64::MAX);
        let mut r = Reader::new(w.as_slice()).unwrap();
        assert!(r.bytes().is_err());
    }

    #[test]
    fn invalid_bool_rejected() {
        let mut w = Writer::new();
        w.u8(7);
        let mut r = Reader::new(w.as_slice()).unwrap();
        assert!(r.bool().is_err());
    }
}
