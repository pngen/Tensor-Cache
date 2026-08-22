#![allow(unsafe_code)]
//! Dynamic loader for the CUDA runtime API. Loads cudart at runtime and
//! resolves the handful of symbols the backend needs. The module handle is
//! held as an integer so the resulting API is Send + Sync.

use std::ffi::{c_char, c_void, CStr, CString};

use tensorcache::error::Error;

/// Dynamically loaded CUDA runtime API.
#[derive(Clone)]
pub struct CudaApi {
    pub get_device_count: unsafe extern "C" fn(*mut i32) -> i32,
    pub set_device: unsafe extern "C" fn(i32) -> i32,
    pub malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    pub memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    pub free: unsafe extern "C" fn(*mut c_void) -> i32,
    pub memset: unsafe extern "C" fn(*mut c_void, i32, usize) -> i32,
    pub get_error_string: unsafe extern "C" fn(i32) -> *const c_char,
    _module_handle: usize,
}

impl CudaApi {
    /// Build an error message from a CUDA error code.
    pub fn err(&self, code: i32, what: &str) -> Error {
        let msg = if code != 0 {
            let ptr = unsafe { (self.get_error_string)(code) };
            if !ptr.is_null() {
                let s = unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned();
                format!("{what}: {s} (code {code})")
            } else {
                format!("{what}: CUDA error {code}")
            }
        } else {
            format!("{what}: unknown error")
        };
        Error::Backend(msg)
    }
}

/// Attempt to load the CUDA runtime and resolve the required symbols.
pub fn load() -> Option<CudaApi> {
    let handle = platform::try_load()?;
    unsafe {
        let load = |sym: &str| platform::symbol(handle, sym).map(|p| p as usize);
        Some(CudaApi {
            get_device_count: transmute_fn(load("cudaGetDeviceCount")?),
            set_device: transmute_fn(load("cudaSetDevice")?),
            malloc: transmute_fn(load("cudaMalloc")?),
            memcpy: transmute_fn(load("cudaMemcpy")?),
            free: transmute_fn(load("cudaFree")?),
            memset: transmute_fn(load("cudaMemset")?),
            get_error_string: transmute_fn(load("cudaGetErrorString")?),
            _module_handle: handle as usize,
        })
    }
}

unsafe fn transmute_fn<T>(p: usize) -> T {
    std::mem::transmute_copy::<usize, T>(&p)
}

#[cfg(windows)]
mod platform {
    use super::*;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(h: *mut c_void, name: *const i8) -> *mut c_void;
    }

    pub fn try_load() -> Option<*mut c_void> {
        const NAMES: [&str; 6] = [
            "cudart64_13.dll",
            "cudart64_12.dll",
            "cudart64_11.dll",
            "cudart64_10.dll",
            "cudart64_9.dll",
            "cudart64.dll",
        ];
        for name in NAMES {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            unsafe {
                let h = LoadLibraryW(wide.as_ptr());
                if !h.is_null() {
                    return Some(h);
                }
            }
        }
        None
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = CString::new(name).ok()?;
        unsafe {
            let p = GetProcAddress(handle, c.as_ptr());
            (!p.is_null()).then_some(p)
        }
    }
}

#[cfg(unix)]
mod platform {
    use super::*;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    pub fn try_load() -> Option<*mut c_void> {
        const NAMES: [&str; 4] = [
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.so",
        ];
        let flags = 0x1 | 0x2; // RTLD_LAZY | RTLD_LOCAL
        for name in NAMES {
            if let Ok(c) = CString::new(name) {
                unsafe {
                    let h = dlopen(c.as_ptr(), flags);
                    if !h.is_null() {
                        return Some(h);
                    }
                }
            }
        }
        None
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = CString::new(name).ok()?;
        unsafe {
            let p = dlsym(handle, c.as_ptr());
            (!p.is_null()).then_some(p)
        }
    }
}
