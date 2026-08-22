#![forbid(unsafe_code)]
//! Bounded framed wire protocol for the Tensor Cache runtime.
//!
//! Every message is carried in a frame with an explicit magic, a version, a
//! message-type tag, a 32-bit length and a CRC-32C checksum over the framing
//! header prefix and the payload. The reader never allocates based on a
//! peer-controlled length beyond a hard cap, rejects unknown magic/version and
//! malformed lengths, and reads partial writes correctly.
//!
//! Two transports use this codec: the coordinator protocol (node to/from the
//! coordinator) and the peer protocol (node to node for object transfer).

use std::io::{Read, Write};
use std::net::TcpStream;

use crate::crc::crc32c;
use crate::error::{Error, Result};
use crate::wire::{Reader, Writer};

/// Frame magic "TCBF" (Tensor Cache Binary Frame).
pub const MAGIC: u32 = 0x5443_4246;
/// Protocol version.
pub const VERSION: u8 = 1;
/// Hard cap on a single frame payload (64 MiB). Larger frames are rejected
/// before allocation.
pub const MAX_FRAME: u32 = 64 * 1024 * 1024;
/// Header size in bytes: magic(4) version(1) type(1) length(4) crc(4) reserved(2).
pub const HEADER_LEN: usize = 16;

/// Message type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MsgType {
    Register = 1,
    Hello = 2,
    Lookup = 3,
    LookupResult = 4,
    Create = 5,
    CreateAck = 6,
    LeaseRenew = 7,
    LeaseGrant = 8,
    Fetch = 9,
    FetchReply = 10,
    Store = 11,
    StoreAck = 12,
    Migrate = 13,
    MigrateAck = 14,
    Heartbeat = 15,
    Error = 16,
}

impl MsgType {
    pub fn tag(self) -> u8 {
        self as u8
    }
    pub fn from_tag(t: u8) -> Result<MsgType> {
        Ok(match t {
            1 => MsgType::Register,
            2 => MsgType::Hello,
            3 => MsgType::Lookup,
            4 => MsgType::LookupResult,
            5 => MsgType::Create,
            6 => MsgType::CreateAck,
            7 => MsgType::LeaseRenew,
            8 => MsgType::LeaseGrant,
            9 => MsgType::Fetch,
            10 => MsgType::FetchReply,
            11 => MsgType::Store,
            12 => MsgType::StoreAck,
            13 => MsgType::Migrate,
            14 => MsgType::MigrateAck,
            15 => MsgType::Heartbeat,
            16 => MsgType::Error,
            _ => return Err(Error::Protocol(format!("unknown message type {t}"))),
        })
    }
}

/// A decoded network message.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    Register {
        node_id: String,
        addr: String,
    },
    Hello {
        epoch: u64,
        boot_id: String,
        node_id: String,
        addr: String,
        lease_ns: u64,
    },
    Lookup {
        namespace: String,
        key: String,
        generation: u64,
        compat: Vec<u8>,
    },
    LookupResult {
        found: bool,
        owner: Option<String>,
        owner_addr: Option<String>,
        generation: u64,
    },
    Create {
        namespace: String,
        key: String,
        generation: u64,
        byte_len: u64,
        compat: Vec<u8>,
        node_id: String,
    },
    CreateAck {
        object_id: String,
        epoch: u64,
        fence: u64,
        owner: String,
    },
    LeaseRenew {
        object_id: String,
        fence: u64,
    },
    LeaseGrant {
        object_id: String,
        epoch: u64,
        fence: u64,
        expires_ns: u64,
    },
    Fetch {
        object_id: String,
        compat: Vec<u8>,
    },
    FetchReply {
        object_id: String,
        data: Vec<u8>,
        crc: u32,
    },
    Store {
        namespace: String,
        key: String,
        generation: u64,
        data: Vec<u8>,
        crc: u32,
        compat: Vec<u8>,
        source: String,
    },
    StoreAck {
        object_id: String,
    },
    Migrate {
        object_id: String,
        new_owner: String,
        new_owner_addr: String,
        fence: u64,
    },
    MigrateAck {
        object_id: String,
        new_owner: String,
        fence: u64,
    },
    Heartbeat {
        node_id: String,
        epoch: u64,
    },
    Error {
        code: String,
        message: String,
    },
}

impl Message {
    pub fn msg_type(&self) -> MsgType {
        match self {
            Message::Register { .. } => MsgType::Register,
            Message::Hello { .. } => MsgType::Hello,
            Message::Lookup { .. } => MsgType::Lookup,
            Message::LookupResult { .. } => MsgType::LookupResult,
            Message::Create { .. } => MsgType::Create,
            Message::CreateAck { .. } => MsgType::CreateAck,
            Message::LeaseRenew { .. } => MsgType::LeaseRenew,
            Message::LeaseGrant { .. } => MsgType::LeaseGrant,
            Message::Fetch { .. } => MsgType::Fetch,
            Message::FetchReply { .. } => MsgType::FetchReply,
            Message::Store { .. } => MsgType::Store,
            Message::StoreAck { .. } => MsgType::StoreAck,
            Message::Migrate { .. } => MsgType::Migrate,
            Message::MigrateAck { .. } => MsgType::MigrateAck,
            Message::Heartbeat { .. } => MsgType::Heartbeat,
            Message::Error { .. } => MsgType::Error,
        }
    }

    /// Encode the message body (without the frame header).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Message::Register { node_id, addr } => {
                w.str(node_id);
                w.str(addr);
            }
            Message::Hello {
                epoch,
                boot_id,
                node_id,
                addr,
                lease_ns,
            } => {
                w.u64(*epoch);
                w.str(boot_id);
                w.str(node_id);
                w.str(addr);
                w.u64(*lease_ns);
            }
            Message::Lookup {
                namespace,
                key,
                generation,
                compat,
            } => {
                w.str(namespace);
                w.str(key);
                w.u64(*generation);
                w.bytes(compat);
            }
            Message::LookupResult {
                found,
                owner,
                owner_addr,
                generation,
            } => {
                w.bool(*found);
                w.u64(*generation);
                match owner {
                    Some(o) => {
                        w.bool(true);
                        w.str(o);
                    }
                    None => {
                        w.bool(false);
                    }
                }
                match owner_addr {
                    Some(a) => {
                        w.bool(true);
                        w.str(a);
                    }
                    None => {
                        w.bool(false);
                    }
                }
            }
            Message::Create {
                namespace,
                key,
                generation,
                byte_len,
                compat,
                node_id,
            } => {
                w.str(namespace);
                w.str(key);
                w.u64(*generation);
                w.u64(*byte_len);
                w.bytes(compat);
                w.str(node_id);
            }
            Message::CreateAck {
                object_id,
                epoch,
                fence,
                owner,
            } => {
                w.str(object_id);
                w.u64(*epoch);
                w.u64(*fence);
                w.str(owner);
            }
            Message::LeaseRenew { object_id, fence } => {
                w.str(object_id);
                w.u64(*fence);
            }
            Message::LeaseGrant {
                object_id,
                epoch,
                fence,
                expires_ns,
            } => {
                w.str(object_id);
                w.u64(*epoch);
                w.u64(*fence);
                w.u64(*expires_ns);
            }
            Message::Fetch { object_id, compat } => {
                w.str(object_id);
                w.bytes(compat);
            }
            Message::FetchReply {
                object_id,
                data,
                crc,
            } => {
                w.str(object_id);
                w.bytes(data);
                w.u32(*crc);
            }
            Message::Store {
                namespace,
                key,
                generation,
                data,
                crc,
                compat,
                source,
            } => {
                w.str(namespace);
                w.str(key);
                w.u64(*generation);
                w.bytes(data);
                w.u32(*crc);
                w.bytes(compat);
                w.str(source);
            }
            Message::StoreAck { object_id } => {
                w.str(object_id);
            }
            Message::Migrate {
                object_id,
                new_owner,
                new_owner_addr,
                fence,
            } => {
                w.str(object_id);
                w.str(new_owner);
                w.str(new_owner_addr);
                w.u64(*fence);
            }
            Message::MigrateAck {
                object_id,
                new_owner,
                fence,
            } => {
                w.str(object_id);
                w.str(new_owner);
                w.u64(*fence);
            }
            Message::Heartbeat { node_id, epoch } => {
                w.str(node_id);
                w.u64(*epoch);
            }
            Message::Error { code, message } => {
                w.str(code);
                w.str(message);
            }
        }
        w.into_inner()
    }

    /// Decode a message body from a type tag.
    pub fn decode(msg_type: MsgType, payload: &[u8]) -> Result<Message> {
        let mut r = Reader::new(payload)?;
        let m = match msg_type {
            MsgType::Register => Message::Register {
                node_id: r.str()?.to_owned(),
                addr: r.str()?.to_owned(),
            },
            MsgType::Hello => Message::Hello {
                epoch: r.u64()?,
                boot_id: r.str()?.to_owned(),
                node_id: r.str()?.to_owned(),
                addr: r.str()?.to_owned(),
                lease_ns: r.u64()?,
            },
            MsgType::Lookup => Message::Lookup {
                namespace: r.str()?.to_owned(),
                key: r.str()?.to_owned(),
                generation: r.u64()?,
                compat: r.bytes()?.to_vec(),
            },
            MsgType::LookupResult => {
                let found = r.bool()?;
                let generation = r.u64()?;
                let has_owner = r.bool()?;
                let owner = if has_owner {
                    Some(r.str()?.to_owned())
                } else {
                    None
                };
                let has_addr = r.bool()?;
                let owner_addr = if has_addr {
                    Some(r.str()?.to_owned())
                } else {
                    None
                };
                Message::LookupResult {
                    found,
                    owner,
                    owner_addr,
                    generation,
                }
            }
            MsgType::Create => Message::Create {
                namespace: r.str()?.to_owned(),
                key: r.str()?.to_owned(),
                generation: r.u64()?,
                byte_len: r.u64()?,
                compat: r.bytes()?.to_vec(),
                node_id: r.str()?.to_owned(),
            },
            MsgType::CreateAck => Message::CreateAck {
                object_id: r.str()?.to_owned(),
                epoch: r.u64()?,
                fence: r.u64()?,
                owner: r.str()?.to_owned(),
            },
            MsgType::LeaseRenew => Message::LeaseRenew {
                object_id: r.str()?.to_owned(),
                fence: r.u64()?,
            },
            MsgType::LeaseGrant => Message::LeaseGrant {
                object_id: r.str()?.to_owned(),
                epoch: r.u64()?,
                fence: r.u64()?,
                expires_ns: r.u64()?,
            },
            MsgType::Fetch => Message::Fetch {
                object_id: r.str()?.to_owned(),
                compat: r.bytes()?.to_vec(),
            },
            MsgType::FetchReply => Message::FetchReply {
                object_id: r.str()?.to_owned(),
                data: r.bytes()?.to_vec(),
                crc: r.u32()?,
            },
            MsgType::Store => Message::Store {
                namespace: r.str()?.to_owned(),
                key: r.str()?.to_owned(),
                generation: r.u64()?,
                data: r.bytes()?.to_vec(),
                crc: r.u32()?,
                compat: r.bytes()?.to_vec(),
                source: r.str()?.to_owned(),
            },
            MsgType::StoreAck => Message::StoreAck {
                object_id: r.str()?.to_owned(),
            },
            MsgType::Migrate => Message::Migrate {
                object_id: r.str()?.to_owned(),
                new_owner: r.str()?.to_owned(),
                new_owner_addr: r.str()?.to_owned(),
                fence: r.u64()?,
            },
            MsgType::MigrateAck => Message::MigrateAck {
                object_id: r.str()?.to_owned(),
                new_owner: r.str()?.to_owned(),
                fence: r.u64()?,
            },
            MsgType::Heartbeat => Message::Heartbeat {
                node_id: r.str()?.to_owned(),
                epoch: r.u64()?,
            },
            MsgType::Error => Message::Error {
                code: r.str()?.to_owned(),
                message: r.str()?.to_owned(),
            },
        };
        if !r.eof() {
            return Err(Error::Protocol("trailing bytes in message".into()));
        }
        Ok(m)
    }

    /// Encode a full frame: 16-byte header plus payload.
    pub fn encode_frame(&self) -> Vec<u8> {
        let body = self.encode();
        encode_frame(self.msg_type().tag(), &body)
    }
}

/// Encode a frame from a message type tag and payload.
pub fn encode_frame(msg_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut head = [0u8; HEADER_LEN];
    head[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    head[4] = VERSION;
    head[5] = msg_type;
    head[6..10].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    // CRC covers the header prefix (0..10) and the payload.
    let mut crc_input = Vec::with_capacity(10 + payload.len());
    crc_input.extend_from_slice(&head[0..10]);
    crc_input.extend_from_slice(payload);
    let crc = crc32c(&crc_input);
    head[10..14].copy_from_slice(&crc.to_le_bytes());
    let mut out = vec![0u8; HEADER_LEN + payload.len()];
    out[..HEADER_LEN].copy_from_slice(&head);
    out[HEADER_LEN..].copy_from_slice(payload);
    out
}

/// Read exactly one frame from a stream, handling partial reads correctly.
/// Returns `Ok(None)` when the peer closes cleanly at a frame boundary.
pub fn read_frame(stream: &mut TcpStream) -> Result<Option<(MsgType, Vec<u8>)>> {
    let mut head = [0u8; HEADER_LEN];
    let n = read_exact_or_eof(stream, &mut head)?;
    if n == 0 {
        return Ok(None);
    }
    if n < HEADER_LEN {
        return Err(Error::Protocol("truncated frame header".into()));
    }
    let magic = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
    if magic != MAGIC {
        return Err(Error::Protocol("bad frame magic".into()));
    }
    if head[4] != VERSION {
        return Err(Error::Protocol(format!("bad frame version {}", head[4])));
    }
    let msg_type = MsgType::from_tag(head[5])?;
    let length = u32::from_le_bytes([head[6], head[7], head[8], head[9]]);
    if length > MAX_FRAME {
        return Err(Error::Protocol(format!(
            "frame length {length} exceeds max {MAX_FRAME}"
        )));
    }
    let mut payload = vec![0u8; length as usize];
    read_exact(stream, &mut payload)?;
    let expected_crc = u32::from_le_bytes([head[10], head[11], head[12], head[13]]);
    let mut crc_input = Vec::with_capacity(10 + payload.len());
    crc_input.extend_from_slice(&head[0..10]);
    crc_input.extend_from_slice(&payload);
    let actual_crc = crc32c(&crc_input);
    if actual_crc != expected_crc {
        return Err(Error::Protocol("frame CRC mismatch".into()));
    }
    Ok(Some((msg_type, payload)))
}

/// Write a full frame to a stream, handling partial writes.
pub fn write_frame(stream: &mut TcpStream, msg: &Message) -> Result<()> {
    let frame = msg.encode_frame();
    stream.write_all(&frame)?;
    stream.flush()?;
    Ok(())
}

fn read_exact_or_eof(stream: &mut TcpStream, buf: &mut [u8]) -> Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        match stream.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e.to_string())),
        }
    }
    Ok(total)
}

fn read_exact(stream: &mut TcpStream, buf: &mut [u8]) -> Result<()> {
    let mut total = 0;
    while total < buf.len() {
        match stream.read(&mut buf[total..]) {
            Ok(0) => return Err(Error::Protocol("connection closed mid-frame".into())),
            Ok(n) => total += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e.to_string())),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(m: &Message) -> Message {
        let frame = m.encode_frame();
        // Simulate a stream over the frame bytes.
        let mut stream = socket_pair_with(&frame);
        let (t, payload) = read_frame(&mut stream).unwrap().unwrap();
        assert_eq!(t, m.msg_type());
        Message::decode(t, &payload).unwrap()
    }

    fn socket_pair_with(bytes: &[u8]) -> TcpStream {
        // Use a loopback pair so read_frame exercises TcpStream semantics.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut writer = std::net::TcpStream::connect(addr).unwrap();
        let reader = listener.accept().unwrap().0;
        writer.write_all(bytes).unwrap();
        writer.flush().unwrap();
        // Drop writer so reader sees EOF after payload.
        drop(writer);
        reader
    }

    #[test]
    fn frame_roundtrip_all_messages() {
        for m in [
            Message::Register {
                node_id: "n1".into(),
                addr: "127.0.0.1:9001".into(),
            },
            Message::Hello {
                epoch: 3,
                boot_id: "boot".into(),
                node_id: "n1".into(),
                addr: "127.0.0.1:9001".into(),
                lease_ns: 1000,
            },
            Message::Lookup {
                namespace: "ns".into(),
                key: "k".into(),
                generation: 1,
                compat: vec![1, 2, 3],
            },
            Message::LookupResult {
                found: true,
                owner: Some("n1".into()),
                owner_addr: Some("127.0.0.1:9001".into()),
                generation: 1,
            },
            Message::Create {
                namespace: "ns".into(),
                key: "k".into(),
                generation: 1,
                byte_len: 8,
                compat: vec![1, 2, 3],
                node_id: "n1".into(),
            },
            Message::CreateAck {
                object_id: "abc".into(),
                epoch: 3,
                fence: 0,
                owner: "n1".into(),
            },
            Message::Fetch {
                object_id: "abc".into(),
                compat: vec![1, 2, 3],
            },
            Message::FetchReply {
                object_id: "abc".into(),
                data: vec![1, 2, 3],
                crc: 12345,
            },
            Message::Store {
                namespace: "ns".into(),
                key: "k".into(),
                generation: 1,
                data: vec![1, 2],
                crc: 99,
                compat: vec![1],
                source: "n1".into(),
            },
            Message::StoreAck {
                object_id: "abc".into(),
            },
            Message::Migrate {
                object_id: "abc".into(),
                new_owner: "n2".into(),
                new_owner_addr: "127.0.0.1:9002".into(),
                fence: 5,
            },
            Message::MigrateAck {
                object_id: "abc".into(),
                new_owner: "n2".into(),
                fence: 5,
            },
            Message::Heartbeat {
                node_id: "n1".into(),
                epoch: 3,
            },
            Message::Error {
                code: "auth".into(),
                message: "detail".into(),
            },
        ] {
            assert_eq!(roundtrip(&m), m);
        }
    }

    #[test]
    fn rejects_bad_magic_and_oversize_length() {
        let mut frame = Message::Hello {
            epoch: 1,
            boot_id: "b".into(),
            node_id: "n".into(),
            addr: "127.0.0.1:1".into(),
            lease_ns: 1,
        }
        .encode_frame();
        frame[0] ^= 0xFF;
        let mut s = socket_pair_with(&frame);
        assert!(read_frame(&mut s).is_err());

        // Oversize length must be rejected before allocation.
        let mut head = [0u8; HEADER_LEN];
        head[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        head[4] = VERSION;
        head[5] = MsgType::Heartbeat.tag();
        head[6..10].copy_from_slice(&(MAX_FRAME + 1).to_le_bytes());
        let bad = head.to_vec();
        // pad so read_frame gets a full header
        let mut s2 = socket_pair_with(&bad);
        assert!(read_frame(&mut s2).is_err());
    }

    #[test]
    fn bad_version_rejected() {
        let mut frame = Message::Hello {
            epoch: 1,
            boot_id: "b".into(),
            node_id: "n".into(),
            addr: "127.0.0.1:1".into(),
            lease_ns: 1,
        }
        .encode_frame();
        frame[4] = 99;
        let mut s = socket_pair_with(&frame);
        assert!(read_frame(&mut s).is_err());
    }
}
