#![forbid(unsafe_code)]
//! Backend abstraction for accelerator devices.
//!
//! The core depends only on this trait and never on a specific vendor. The
//! mandatory CPU backend is fully implemented here. An optional CUDA backend
//! is provided by the separate tensorcache-cuda crate behind the cuda feature
//! and loaded dynamically so the core has no link-time dependency. HIP, Level
//! Zero, Metal and Vulkan backends may be added later by implementing this
//! trait; none of them are claimed as implemented today.

use crate::error::Result;

/// A backend identifier: a backend name plus a zero-based device index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendId {
    pub name: String,
    pub device: usize,
}

impl BackendId {
    pub fn new(name: impl Into<String>, device: usize) -> Self {
        BackendId {
            name: name.into(),
            device,
        }
    }

    pub fn cpu(device: usize) -> Self {
        BackendId::new("cpu", device)
    }

    pub fn cuda(device: usize) -> Self {
        BackendId::new("cuda", device)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.name, self.device)
    }
}

/// The opaque storage handle of a device allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceHandle {
    /// A CPU-resident byte buffer (used by the mandatory CPU backend).
    Cpu(Vec<u8>),
    /// An opaque backend-internal device id, e.g. a CUDA pointer address.
    Opaque(u64),
}

/// A live device allocation owned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBuffer {
    pub backend: BackendId,
    pub bytes: usize,
    pub handle: DeviceHandle,
}

impl DeviceBuffer {
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

/// The contract every device backend must satisfy.
pub trait Backend: Send + Sync {
    /// The backend identifier (name and device index).
    fn id(&self) -> &BackendId;

    /// Nominal device byte capacity reported to accounting.
    fn byte_capacity(&self) -> u64;

    /// Bytes currently allocated on this device (used for leak detection).
    fn bytes_allocated(&self) -> u64;

    /// Allocate a zero-initialized device buffer of the given byte size.
    fn allocate(&self, bytes: usize) -> Result<DeviceBuffer>;

    /// Copy host bytes into a device buffer (H2D), validating bounds.
    fn to_device(&self, host: &[u8], dev: &mut DeviceBuffer) -> Result<()>;

    /// Copy device bytes back into a host buffer (D2H), validating bounds.
    fn device_to_host(&self, dev: &DeviceBuffer, host: &mut [u8]) -> Result<()>;

    /// Copy a host slice into a device buffer at a byte offset.
    fn copy_in(&self, dev: &mut DeviceBuffer, offset: usize, data: &[u8]) -> Result<()>;

    /// Copy device bytes out at a byte offset into a host slice.
    fn copy_out(&self, dev: &DeviceBuffer, offset: usize, data: &mut [u8]) -> Result<()>;

    /// Fill a prefix of a device buffer with a byte value.
    fn fill(&self, dev: &mut DeviceBuffer, bytes: usize, value: u8) -> Result<()>;

    /// Release a device buffer, returning it to the backend pool/free list.
    fn free(&self, dev: DeviceBuffer) -> Result<()>;
}

/// A registry of live backends.
#[derive(Default)]
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        BackendRegistry {
            backends: Vec::new(),
        }
    }

    pub fn register(&mut self, b: Box<dyn Backend>) {
        self.backends.push(b);
    }

    pub fn get(&self, id: &BackendId) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.id() == id)
            .map(|b| b.as_ref())
    }

    pub fn count(&self) -> usize {
        self.backends.len()
    }
}
