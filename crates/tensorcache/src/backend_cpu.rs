#![forbid(unsafe_code)]
//! The mandatory CPU backend.
//!
//! It models a device allocation as a plain host byte buffer. This satisfies
//! the same backend contract as a real accelerator (allocate, H2D copy, D2H
//! copy, fill, free) so the promote/demote/persist/verify machinery exercises
//! real code paths even without accelerator hardware.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::{Backend, BackendId, DeviceBuffer, DeviceHandle};
use crate::error::{Error, Result};

/// A CPU-resident device backend.
pub struct CpuBackend {
    id: BackendId,
    capacity: u64,
    allocated: AtomicU64,
}

impl CpuBackend {
    pub fn new(device: usize, capacity: u64) -> Self {
        CpuBackend {
            id: BackendId::cpu(device),
            capacity,
            allocated: AtomicU64::new(0),
        }
    }
}

impl Backend for CpuBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn byte_capacity(&self) -> u64 {
        self.capacity
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated.load(Ordering::SeqCst)
    }

    fn allocate(&self, bytes: usize) -> Result<DeviceBuffer> {
        let before = self.allocated.load(Ordering::SeqCst);
        let after = before
            .checked_add(bytes as u64)
            .ok_or_else(|| Error::Backend("CPU device byte accounting overflow".into()))?;
        if after > self.capacity {
            return Err(Error::Backend(format!(
                "CPU device allocation of {bytes} bytes would exceed capacity {}",
                self.capacity
            )));
        }
        let v = vec![0u8; bytes];
        self.allocated.store(after, Ordering::SeqCst);
        Ok(DeviceBuffer {
            backend: self.id.clone(),
            bytes,
            handle: DeviceHandle::Cpu(v),
        })
    }

    fn to_device(&self, host: &[u8], dev: &mut DeviceBuffer) -> Result<()> {
        if host.len() != dev.bytes {
            return Err(Error::Backend(format!(
                "H2D length mismatch: host {} vs device {}",
                host.len(),
                dev.bytes
            )));
        }
        match &mut dev.handle {
            DeviceHandle::Cpu(v) => v.copy_from_slice(host),
            _ => return Err(Error::Backend("not a CPU device buffer".into())),
        }
        Ok(())
    }

    fn device_to_host(&self, dev: &DeviceBuffer, host: &mut [u8]) -> Result<()> {
        if host.len() != dev.bytes {
            return Err(Error::Backend(format!(
                "D2H length mismatch: host {} vs device {}",
                host.len(),
                dev.bytes
            )));
        }
        match &dev.handle {
            DeviceHandle::Cpu(v) => host.copy_from_slice(v),
            _ => return Err(Error::Backend("not a CPU device buffer".into())),
        }
        Ok(())
    }

    fn copy_in(&self, dev: &mut DeviceBuffer, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::Backend("copy_in offset overflow".into()))?;
        if end > dev.bytes {
            return Err(Error::Backend(format!(
                "copy_in out of bounds: {offset}+{} > {}",
                data.len(),
                dev.bytes
            )));
        }
        match &mut dev.handle {
            DeviceHandle::Cpu(v) => v[offset..end].copy_from_slice(data),
            _ => return Err(Error::Backend("not a CPU device buffer".into())),
        }
        Ok(())
    }

    fn copy_out(&self, dev: &DeviceBuffer, offset: usize, data: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::Backend("copy_out offset overflow".into()))?;
        if end > dev.bytes {
            return Err(Error::Backend(format!(
                "copy_out out of bounds: {offset}+{} > {}",
                data.len(),
                dev.bytes
            )));
        }
        match &dev.handle {
            DeviceHandle::Cpu(v) => data.copy_from_slice(&v[offset..end]),
            _ => return Err(Error::Backend("not a CPU device buffer".into())),
        }
        Ok(())
    }

    fn fill(&self, dev: &mut DeviceBuffer, bytes: usize, value: u8) -> Result<()> {
        if bytes > dev.bytes {
            return Err(Error::Backend(format!(
                "fill exceeds device size {bytes} > {}",
                dev.bytes
            )));
        }
        match &mut dev.handle {
            DeviceHandle::Cpu(v) => v[..bytes].fill(value),
            _ => return Err(Error::Backend("not a CPU device buffer".into())),
        }
        Ok(())
    }

    fn free(&self, dev: DeviceBuffer) -> Result<()> {
        let n = match dev.handle {
            DeviceHandle::Cpu(v) => v.len() as u64,
            _ => return Err(Error::Backend("not a CPU device buffer".into())),
        };
        self.allocated.fetch_sub(n, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_roundtrip_and_accounting() {
        let b = CpuBackend::new(0, 1 << 20);
        assert_eq!(b.bytes_allocated(), 0);
        let mut dev = b.allocate(16).unwrap();
        assert_eq!(b.bytes_allocated(), 16);
        let host = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        b.to_device(&host, &mut dev).unwrap();
        let mut out = vec![0u8; 16];
        b.device_to_host(&dev, &mut out).unwrap();
        assert_eq!(out, host);
        b.free(dev).unwrap();
        assert_eq!(b.bytes_allocated(), 0);
    }

    #[test]
    fn cpu_backend_bounds_enforced() {
        let b = CpuBackend::new(0, 1 << 20);
        let mut dev = b.allocate(8).unwrap();
        assert!(b
            .copy_in(&mut dev, 0, &[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .is_err());
        assert!(b.fill(&mut dev, 9, 0).is_err());
        // Capacity bound.
        let tiny = CpuBackend::new(0, 4);
        assert!(tiny.allocate(8).is_err());
    }
}
