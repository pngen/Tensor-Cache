//! Minimal argument parsing for the Tensor Cache CLI (no external crates).
//!
//! The CLI is intentionally small and hand-parsed. Commands take the form
//! `tensorcache <subcommand> [--flag value ...]`; flags are read as a
//! key/value list.

use std::collections::HashMap;

use tensorcache::dtype::Dtype;
use tensorcache::error::{Error, Result};
use tensorcache::geometry::{Layout, Shape};

/// Parse a command line as a subcommand plus a flag map.
pub fn parse(args: &[String]) -> (String, HashMap<String, String>) {
    let mut flags: HashMap<String, String> = HashMap::new();
    let mut sub = String::new();
    let mut rest: Vec<String> = args.to_vec();
    if !rest.is_empty() {
        sub = rest.remove(0);
    }
    while !rest.is_empty() {
        let tok = rest.remove(0);
        if let Some(stripped) = tok.strip_prefix("--") {
            let key = stripped.to_string();
            if rest.is_empty() {
                flags.insert(key, "true".to_string());
            } else {
                let val = rest.remove(0);
                flags.insert(key, val);
            }
        } else {
            flags.insert("_pos".to_string(), tok);
        }
    }
    (sub, flags)
}

/// Get a required string flag.
pub fn req<'a>(flags: &'a HashMap<String, String>, name: &str) -> Result<&'a str> {
    flags
        .get(name)
        .map(|s| s.as_str())
        .ok_or_else(|| Error::InvalidArgument(format!("missing required --{name}")))
}

/// Get an optional string flag.
pub fn opt<'a>(flags: &'a HashMap<String, String>, name: &str) -> Option<&'a str> {
    flags.get(name).map(|s| s.as_str())
}

/// Parse a u64 flag.
pub fn num(flags: &HashMap<String, String>, name: &str, default: u64) -> Result<u64> {
    match flags.get(name) {
        Some(v) => v
            .parse::<u64>()
            .map_err(|e| Error::InvalidArgument(format!("invalid --{name}: {e}"))),
        None => Ok(default),
    }
}

/// Parse a dtype from its textual name.
pub fn parse_dtype(s: &str) -> Result<Dtype> {
    Ok(match s {
        "f64" => Dtype::F64,
        "f32" => Dtype::F32,
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        "f8" => Dtype::F8,
        "i64" => Dtype::I64,
        "i32" => Dtype::I32,
        "i16" => Dtype::I16,
        "i8" => Dtype::I8,
        "u64" => Dtype::U64,
        "u32" => Dtype::U32,
        "u16" => Dtype::U16,
        "u8" => Dtype::U8,
        "bool" => Dtype::Bool,
        other => return Err(Error::InvalidArgument(format!("unknown dtype {other}"))),
    })
}

/// Parse a shape such as "32x64" or "128".
pub fn parse_shape(s: &str) -> Result<Shape> {
    let dims: Vec<u64> = if s.is_empty() {
        vec![]
    } else {
        s.split('x')
            .map(|p| {
                p.parse::<u64>()
                    .map_err(|e| Error::InvalidArgument(format!("invalid shape dim {p}: {e}")))
            })
            .collect::<Result<Vec<_>>>()?
    };
    Shape::new(dims)
}

/// Parse a layout such as "row" (default), "col", or "strided".
pub fn parse_layout(s: &str) -> Layout {
    match s {
        "col" | "column" => Layout::ColMajor,
        _ => Layout::RowMajor,
    }
}

/// Print a human-readable memory size.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} {}", UNITS[u])
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}
