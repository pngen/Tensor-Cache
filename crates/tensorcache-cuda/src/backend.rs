#![allow(unsafe_code)]
//! CUDA device backend implementing the Tensor Cache backend contract.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use tensorcache::backend::{Backend, BackendId, DeviceBuffer, DeviceHandle};
use tensorcache::error::{Error, Result};

use crate::loader::{load, CudaApi};

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

/// A CUDA device backend.
pub struct CudaBackend {
    id: BackendId,
    capacity: u64,
    allocated: AtomicU64,
    api: CudaApi,
}

impl CudaBackend {
    /// Create a backend for a CUDA device and probe the runtime.
    pub fn new(device: usize, capacity: u64) -> Result<CudaBackend> {
        let api = load().ok_or_else(|| Error::Backend("CUDA runtime library not found".into()))?;
        let mut count: i32 = 0;
        let rc = unsafe { (api.get_device_count)(&mut count) };
        if rc != 0 {
            return Err(api.err(rc, "cudaGetDeviceCount"));
        }
        if count <= 0 {
            return Err(Error::Backend("no CUDA devices present".into()));
        }
        if device >= count as usize {
            return Err(Error::Backend(format!(
                "requested device {device} but only {count} present"
            )));
        }
        let rc = unsafe { (api.set_device)(device as i32) };
        if rc != 0 {
            return Err(api.err(rc, "cudaSetDevice"));
        }
        Ok(CudaBackend {
            id: BackendId::cuda(device),
            capacity,
            allocated: AtomicU64::new(0),
            api,
        })
    }

    /// The number of CUDA devices available, if the runtime is present.
    pub fn device_count() -> Result<i32> {
        let api = load().ok_or_else(|| Error::Backend("CUDA runtime library not found".into()))?;
        let mut count: i32 = 0;
        let rc = unsafe { (api.get_device_count)(&mut count) };
        if rc != 0 {
            return Err(api.err(rc, "cudaGetDeviceCount"));
        }
        Ok(count)
    }
}

fn dev_ptr(dev: &DeviceBuffer) -> Result<*mut c_void> {
    match dev.handle {
        DeviceHandle::Opaque(p) => Ok(p as *mut c_void),
        _ => Err(Error::Backend("not a CUDA device buffer".into())),
    }
}

impl Backend for CudaBackend {
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
        if bytes == 0 {
            return Ok(DeviceBuffer {
                backend: self.id.clone(),
                bytes: 0,
                handle: DeviceHandle::Opaque(0),
            });
        }
        let before = self.allocated.load(Ordering::SeqCst);
        let after = before
            .checked_add(bytes as u64)
            .ok_or_else(|| Error::Backend("CUDA byte accounting overflow".into()))?;
        if after > self.capacity {
            return Err(Error::Backend(format!(
                "CUDA allocation of {bytes} bytes would exceed capacity {}",
                self.capacity
            )));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (self.api.malloc)(&mut ptr, bytes) };
        if rc != 0 || ptr.is_null() {
            return Err(self.api.err(rc, "cudaMalloc"));
        }
        self.allocated.store(after, Ordering::SeqCst);
        Ok(DeviceBuffer {
            backend: self.id.clone(),
            bytes,
            handle: DeviceHandle::Opaque(ptr as u64),
        })
    }

    fn to_device(&self, host: &[u8], dev: &mut DeviceBuffer) -> Result<()> {
        if host.len() != dev.bytes {
            return Err(Error::Backend("H2D length mismatch".into()));
        }
        if dev.bytes == 0 {
            return Ok(());
        }
        let rc = unsafe {
            (self.api.memcpy)(
                dev_ptr(dev)?,
                host.as_ptr() as *const c_void,
                host.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if rc != 0 {
            return Err(self.api.err(rc, "cudaMemcpy(H2D)"));
        }
        Ok(())
    }

    fn device_to_host(&self, dev: &DeviceBuffer, host: &mut [u8]) -> Result<()> {
        if host.len() != dev.bytes {
            return Err(Error::Backend("D2H length mismatch".into()));
        }
        if dev.bytes == 0 {
            return Ok(());
        }
        let rc = unsafe {
            (self.api.memcpy)(
                host.as_mut_ptr() as *mut c_void,
                dev_ptr(dev)?,
                host.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != 0 {
            return Err(self.api.err(rc, "cudaMemcpy(D2H)"));
        }
        Ok(())
    }

    fn copy_in(&self, dev: &mut DeviceBuffer, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::Backend("copy_in offset overflow".into()))?;
        if end > dev.bytes {
            return Err(Error::Backend("copy_in out of bounds".into()));
        }
        if data.is_empty() {
            return Ok(());
        }
        let base = dev_ptr(dev)? as *mut u8;
        let dst = unsafe { base.add(offset) } as *mut c_void;
        let rc = unsafe {
            (self.api.memcpy)(
                dst,
                data.as_ptr() as *const c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if rc != 0 {
            return Err(self.api.err(rc, "cudaMemcpy(copy_in)"));
        }
        Ok(())
    }

    fn copy_out(&self, dev: &DeviceBuffer, offset: usize, data: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::Backend("copy_out offset overflow".into()))?;
        if end > dev.bytes {
            return Err(Error::Backend("copy_out out of bounds".into()));
        }
        if data.is_empty() {
            return Ok(());
        }
        let base = dev_ptr(dev)? as *const u8;
        let src = unsafe { base.add(offset) } as *const c_void;
        let rc = unsafe {
            (self.api.memcpy)(
                data.as_mut_ptr() as *mut c_void,
                src,
                data.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != 0 {
            return Err(self.api.err(rc, "cudaMemcpy(copy_out)"));
        }
        Ok(())
    }

    fn fill(&self, dev: &mut DeviceBuffer, bytes: usize, value: u8) -> Result<()> {
        if bytes > dev.bytes {
            return Err(Error::Backend("fill exceeds device size".into()));
        }
        if bytes == 0 {
            return Ok(());
        }
        let rc = unsafe { (self.api.memset)(dev_ptr(dev)?, value as i32, bytes) };
        if rc != 0 {
            return Err(self.api.err(rc, "cudaMemset"));
        }
        Ok(())
    }

    fn free(&self, dev: DeviceBuffer) -> Result<()> {
        if dev.bytes == 0 {
            return Ok(());
        }
        let ptr = match dev.handle {
            DeviceHandle::Opaque(p) => p as *mut c_void,
            _ => return Err(Error::Backend("not a CUDA buffer".into())),
        };
        let rc = unsafe { (self.api.free)(ptr) };
        if rc != 0 {
            return Err(self.api.err(rc, "cudaFree"));
        }
        self.allocated.fetch_sub(dev.bytes as u64, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_roundtrip_and_accounting() {
        let b = match CudaBackend::new(0, 1 << 30) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("CUDA not available ({e}); skipping real validation");
                return;
            }
        };
        assert!(b.byte_capacity() >= (1 << 30));
        let mut dev = b.allocate(4096).unwrap();
        assert_eq!(b.bytes_allocated(), 4096);
        let host: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        b.to_device(&host, &mut dev).unwrap();
        let mut out = vec![0u8; 4096];
        b.device_to_host(&dev, &mut out).unwrap();
        assert_eq!(out, host);
        b.fill(&mut dev, 4096, 0xAB).unwrap();
        let mut filled = vec![0u8; 4096];
        b.device_to_host(&dev, &mut filled).unwrap();
        assert!(filled.iter().all(|&x| x == 0xAB));
        b.free(dev).unwrap();
        assert_eq!(b.bytes_allocated(), 0);
    }

    #[test]
    fn repeated_allocations_release_exactly() {
        let b = match CudaBackend::new(0, 1 << 30) {
            Ok(b) => b,
            Err(_) => return,
        };
        for _ in 0..32 {
            let mut dev = b.allocate(1 << 16).unwrap();
            let host = vec![9u8; 1 << 16];
            b.to_device(&host, &mut dev).unwrap();
            let mut out = vec![0u8; 1 << 16];
            b.device_to_host(&dev, &mut out).unwrap();
            assert_eq!(out, host);
            b.free(dev).unwrap();
        }
        assert_eq!(b.bytes_allocated(), 0);
    }
}
