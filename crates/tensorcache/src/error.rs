//! Typed errors for Tensor Cache.
//!
//! Tensor Cache has a strict policy: no panic-based normal error handling.
//! Every recoverable failure is surfaced through this error type and the
//! crate-level `Result` alias.

use std::fmt;

/// The crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// The typed error enum for all Tensor Cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An I/O failure from the host filesystem or a socket.
    Io(String),
    /// An invalid or inconsistent argument supplied by the caller.
    InvalidArgument(String),
    /// A geometry violation: impossible rank, shape, product overflow, or a
    /// byte-length that does not match the declared geometry.
    Geometry(String),
    /// An incompatible tensor geometry, dtype, layout, model, runtime or
    /// revision prevented a safe reuse.
    Compatibility(String),
    /// The requested object was not present.
    NotFound(String),
    /// An object with the same canonical identity already exists.
    Exists(String),
    /// The requested admission was rejected by policy (capacity, quota, value).
    AdmissionRejected(String),
    /// An integrity failure: a block or manifest checksum did not match.
    Integrity(String),
    /// A durable persistence failure: incomplete commit, malformed manifest or
    /// a corrupt/corrupted on-disk artifact.
    Persistence(String),
    /// A protocol violation: bad magic, bad version, malformed frame, unsized
    /// or impossible length.
    Protocol(String),
    /// An authority failure: stale epoch, stale boot identity, stale
    /// generation, missing/expired lease, or a conflicting writer.
    Authority(String),
    /// An illegal residency-state transition.
    Residency(String),
    /// A deduplication reference-counting or ownership violation.
    Dedup(String),
    /// A backend failure: unavailable device, allocation failure, or transfer
    /// failure.
    Backend(String),
    /// A tensor could not be reconstructed because one or more blocks are
    /// missing or corrupted.
    Reconstruct(String),
    /// A concurrency or synchronization failure.
    Concurrency(String),
    /// A migration failure (interrupted, inconsistent, or dual ownership).
    Migration(String),
    /// A resource-accounting failure: negative, leaked or misaccounted bytes.
    Accounting(String),
    /// Any internal invariant failure that should not have been reachable.
    Internal(String),
}

impl Error {
    /// A short machine-readable category tag, retained for logs and tooling.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Io(_) => "io",
            Error::InvalidArgument(_) => "invalid-argument",
            Error::Geometry(_) => "geometry",
            Error::Compatibility(_) => "compatibility",
            Error::NotFound(_) => "not-found",
            Error::Exists(_) => "exists",
            Error::AdmissionRejected(_) => "admission-rejected",
            Error::Integrity(_) => "integrity",
            Error::Persistence(_) => "persistence",
            Error::Protocol(_) => "protocol",
            Error::Authority(_) => "authority",
            Error::Residency(_) => "residency",
            Error::Dedup(_) => "dedup",
            Error::Backend(_) => "backend",
            Error::Reconstruct(_) => "reconstruct",
            Error::Concurrency(_) => "concurrency",
            Error::Migration(_) => "migration",
            Error::Accounting(_) => "accounting",
            Error::Internal(_) => "internal",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Error::Io(m) => m,
            Error::InvalidArgument(m) => m,
            Error::Geometry(m) => m,
            Error::Compatibility(m) => m,
            Error::NotFound(m) => m,
            Error::Exists(m) => m,
            Error::AdmissionRejected(m) => m,
            Error::Integrity(m) => m,
            Error::Persistence(m) => m,
            Error::Protocol(m) => m,
            Error::Authority(m) => m,
            Error::Residency(m) => m,
            Error::Dedup(m) => m,
            Error::Backend(m) => m,
            Error::Reconstruct(m) => m,
            Error::Concurrency(m) => m,
            Error::Migration(m) => m,
            Error::Accounting(m) => m,
            Error::Internal(m) => m,
        };
        write!(f, "{s}")
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// Build an `Internal` error from a message.
pub fn internal(msg: impl Into<String>) -> Error {
    Error::Internal(msg.into())
}

/// Build an `InvalidArgument` error from a message.
pub fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_kind_is_stable() {
        assert_eq!(Error::NotFound("x".into()).kind(), "not-found");
        assert_eq!(Error::Authority("x".into()).kind(), "authority");
    }
}
