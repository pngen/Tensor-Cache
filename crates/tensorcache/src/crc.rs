#![forbid(unsafe_code)]
//! CRC-32C (Castagnoli) checksum, implemented in pure Rust.
//!
//! The reflected polynomial is 0x1EDC6F41; the bit form used by the
//! table-based update is 0x82F63B78, with initial value 0xFFFFFFFF and a
//! final XOR of 0xFFFFFFFF. This is the CRC variant used by iSCSI, SCTP and
//! Linux ext4 for per-block checksums.

const POLY: u32 = 0x82F6_3B78;

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const TABLE: [u32; 256] = build_table();

/// Compute the CRC-32C of a byte slice.
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_continue(0, data)
}

/// Continue a CRC-32C computation, combining a prior CRC with new bytes.
/// This is useful for streaming or incremental verification.
pub fn crc32c_continue(prior: u32, data: &[u8]) -> u32 {
    let mut crc = prior ^ 0xFFFF_FFFF;
    for &b in data {
        crc = TABLE[((crc ^ b as u32) & 0x00FF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known_vector() {
        // The canonical CRC-32C check value for the string "123456789".
        let c = crc32c(b"123456789");
        assert_eq!(c, 0xE306_9283);
    }

    #[test]
    fn crc32c_empty() {
        assert_eq!(crc32c(b""), 0x0000_0000);
    }

    #[test]
    fn crc32c_streaming_equals_oneshot() {
        let data = b"a streaming integrity payload that is long enough";
        let mut c = 0u32;
        for part in data.chunks(3) {
            c = crc32c_continue(c, part);
        }
        assert_eq!(c, crc32c(data));
    }

    #[test]
    fn crc32c_is_order_sensitive() {
        assert_ne!(crc32c(b"ab"), crc32c(b"ba"));
    }
}
