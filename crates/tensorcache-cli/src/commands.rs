//! Local CLI commands.

use std::collections::HashMap;
use std::path::PathBuf;

use tensorcache::compat::CompatKey;
use tensorcache::error::{Error, Result};
use tensorcache::ident::ObjectId;
use tensorcache::runtime::{RuntimeConfig, TensorCache};
use tensorcache::tiers::Tier;

use crate::args;

pub fn dispatch(sub: &str, flags: &HashMap<String, String>) -> Result<()> {
    match sub {
        "" | "help" | "--help" | "-h" => print_help(),
        "create" => cmd_create(flags),
        "lookup" => cmd_lookup(flags),
        "list" => cmd_list(flags),
        "stats" => cmd_stats(flags),
        "persist" => cmd_object_action(flags, "persist"),
        "promote" => cmd_object_action(flags, "promote"),
        "demote" => cmd_object_action(flags, "demote"),
        "restore" => cmd_restore(flags),
        "evict" => cmd_object_action(flags, "evict"),
        "delete" => cmd_object_action(flags, "delete"),
        "verify" => cmd_verify(flags),
        "replicate" => cmd_object_action(flags, "replicate"),
        "benchmark" => cmd_benchmark(flags),
        "migrate" => crate::server::cmd_migrate(flags),
        "coordinator" => crate::server::run_coordinator(flags),
        "node" => crate::server::run_node(flags),
        other => Err(Error::InvalidArgument(format!(
            "unknown subcommand {other}"
        ))),
    }
}

fn print_help() -> Result<()> {
    println!(
        "Tensor Cache {} - reusable tensor-shaped state runtime",
        env!("CARGO_PKG_VERSION")
    );
    println!("Usage: tensorcache <subcommand> [--flag value ...]");
    println!("  coordinator --listen ADDR [--lease-ns NS] [--snapshot DIR]");
    println!("  node --id ID --listen ADDR --coordinator ADDR --store DIR [--capacity BYTES]");
    println!("  create --store DIR --namespace NS --key K [--generation G] [--dtype D] --shape RXC --file F");
    println!(
        "  lookup --store DIR --namespace NS --key K [--generation G] [--dtype D] --shape RXC"
    );
    println!("  list / stats --store DIR");
    println!("  persist|promote|demote|evict|delete|verify --store DIR --object OID [--tier TIER]");
    println!("  restore --store DIR --object OID [--tier TIER] [--out FILE]");
    println!("  migrate --node-addr ADDR --object OID --to OWNER --to-addr ADDR [--fence F]");
    println!("  benchmark --store DIR [--objects N] [--bytes B]");
    Ok(())
}

fn open_store(dir: &str, capacity: u64) -> Result<TensorCache> {
    let config = RuntimeConfig {
        host_capacity: capacity.max(1 << 20),
        persistent_path: Some(PathBuf::from(dir)),
        ..Default::default()
    };
    TensorCache::new(config)
}

fn build_compat(flags: &HashMap<String, String>) -> Result<CompatKey> {
    let dtype = match args::opt(flags, "dtype") {
        Some(d) => args::parse_dtype(d)?,
        None => tensorcache::dtype::Dtype::F32,
    };
    let shape = match args::opt(flags, "shape") {
        Some(s) => args::parse_shape(s)?,
        None => tensorcache::geometry::Shape::new(vec![])?,
    };
    let layout = args::parse_layout(args::opt(flags, "layout").unwrap_or("row"));
    let mut compat = CompatKey {
        dtype,
        shape,
        layout,
        ..Default::default()
    };
    if let Some(m) = args::opt(flags, "model") {
        compat.model = Some(m.to_string());
    }
    if let Some(m) = args::opt(flags, "revision") {
        compat.model_revision = Some(m.to_string());
    }
    if let Some(r) = args::opt(flags, "runtime") {
        compat.runtime_version = Some(r.to_string());
    }
    if let Some(o) = args::opt(flags, "op") {
        compat.operation = Some(o.to_string());
    }
    if let Some(p) = args::opt(flags, "precision") {
        compat.precision = Some(p.to_string());
    }
    Ok(compat)
}

fn read_payload(flags: &HashMap<String, String>) -> Result<Vec<u8>> {
    if let Some(f) = args::opt(flags, "file") {
        std::fs::read(f).map_err(|e| Error::Io(e.to_string()))
    } else if let Some(n) = args::opt(flags, "fill") {
        let n = n
            .parse::<usize>()
            .map_err(|e| Error::InvalidArgument(format!("bad --fill: {e}")))?;
        Ok((0..n).map(|i| (i % 251) as u8).collect())
    } else {
        Err(Error::InvalidArgument("need --file or --fill".into()))
    }
}

fn cmd_create(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let ns = args::req(flags, "namespace")?;
    let key = args::req(flags, "key")?;
    let gen = args::num(flags, "generation", 0)?;
    let compat = build_compat(flags)?;
    let payload = read_payload(flags)?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    let oid = tc.register(ns, key, gen, compat, &payload)?;
    tc.persist(&oid)?;
    println!("created object {}", oid.to_hex());
    println!("  namespace={ns} key={key} generation={gen}");
    println!("  bytes={}", payload.len());
    println!("  compat_id={}", tc.entry_compat_id(&oid)?);
    Ok(())
}

fn cmd_lookup(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let ns = args::req(flags, "namespace")?;
    let key = args::req(flags, "key")?;
    let gen = args::num(flags, "generation", 0)?;
    let compat = build_compat(flags)?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    match tc.lookup(ns, key, gen, &compat) {
        Ok(res) => {
            println!("hit");
            println!("  object_id={}", res.object_id);
            println!(
                "  source_tier={}",
                res.source_tier
                    .map(|t| t.label())
                    .unwrap_or_else(|| "none".into())
            );
            println!("  bytes={}", res.bytes);
            println!(
                "  reconstruction_avoided_ns={}",
                res.reconstruction_avoided_ns
            );
            println!("  rationale={}", res.rationale);
        }
        Err(e) => println!("miss: {e}"),
    }
    Ok(())
}

fn cmd_list(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    let res = tc.resources();
    println!("objects={}", res.object_count);
    Ok(())
}

fn cmd_stats(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    let r = tc.resources();
    println!(
        "host_used={} ({})",
        r.host_used,
        args::human_bytes(r.host_used)
    );
    println!(
        "host_capacity={} ({})",
        r.host_capacity,
        args::human_bytes(r.host_capacity)
    );
    println!("host_reserved={}", r.host_reserved);
    println!("accel_used={}", r.accel_used);
    println!(
        "storage_used={} ({})",
        r.storage_used,
        args::human_bytes(r.storage_used)
    );
    println!("storage_capacity={}", r.storage_capacity);
    println!("objects={}", r.object_count);
    println!("blocks={}", r.block_count);
    println!("replicas={}", r.replica_count);
    Ok(())
}

fn cmd_object_action(flags: &HashMap<String, String>, action: &str) -> Result<()> {
    let store = args::req(flags, "store")?;
    let oid = ObjectId::from_hex(
        args::opt(flags, "object").ok_or_else(|| Error::InvalidArgument("need --object".into()))?,
    )?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    match action {
        "persist" => tc.persist(&oid)?,
        "evict" => tc.evict(&oid)?,
        "delete" => tc.delete(&oid)?,
        "replicate" => {
            let tier = parse_tier(args::opt(flags, "tier").unwrap_or("host"))?;
            tc.replicate(&oid, &tier)?;
        }
        "promote" => {
            let tier = parse_tier(args::opt(flags, "tier").unwrap_or("host"))?;
            tc.promote(&oid, &tier)?;
        }
        "demote" => {
            let tier = parse_tier(args::opt(flags, "tier").unwrap_or("accelerator"))?;
            tc.demote(&oid, &tier)?;
        }
        _ => return Err(Error::InvalidArgument("unknown action".into())),
    }
    println!("{action} ok for {oid}");
    Ok(())
}

fn cmd_restore(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let oid = ObjectId::from_hex(
        args::opt(flags, "object").ok_or_else(|| Error::InvalidArgument("need --object".into()))?,
    )?;
    let tier = parse_tier(args::opt(flags, "tier").unwrap_or("host"))?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    let bytes = tc.restore(&oid, &tier)?;
    if let Some(out) = args::opt(flags, "out") {
        std::fs::write(out, &bytes)?;
        println!("restored {} bytes to {out}", bytes.len());
    } else {
        println!("restored {} bytes for {oid}", bytes.len());
    }
    Ok(())
}

fn cmd_verify(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let oid = ObjectId::from_hex(
        args::opt(flags, "object").ok_or_else(|| Error::InvalidArgument("need --object".into()))?,
    )?;
    let tc = open_store(store, args::num(flags, "capacity", 1 << 30)?)?;
    let report = tc.verify(&oid)?;
    println!(
        "verify ok: {} placements, {} bytes, clean={}",
        report.checked_placements, report.verified_bytes, report.clean
    );
    Ok(())
}

fn cmd_benchmark(flags: &HashMap<String, String>) -> Result<()> {
    let store = args::req(flags, "store")?;
    let n = args::num(flags, "objects", 2000)? as usize;
    let bytes = args::num(flags, "bytes", 4096)? as usize;
    let tc = open_store(store, 1 << 30)?;
    let compat = CompatKey {
        dtype: tensorcache::dtype::Dtype::F32,
        shape: tensorcache::geometry::Shape::new(vec![bytes as u64 / 4])?,
        layout: tensorcache::geometry::Layout::RowMajor,
        model: Some("bench".into()),
        ..Default::default()
    };
    let payload: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    let _ = tc.register("bench", "seed", 0, compat.clone(), &payload)?;
    let t0 = std::time::Instant::now();
    let mut oids = Vec::with_capacity(n);
    for i in 0..n {
        let o = tc.register("bench", format!("{i}"), 0, compat.clone(), &payload)?;
        oids.push(o);
    }
    let create_elapsed = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut hits = 0u64;
    for o in &oids {
        let meta = tc.metadata(o).unwrap();
        if tc.lookup("bench", &meta.key, 0, &compat).is_ok() {
            hits += 1;
        }
    }
    let lookup_elapsed = t1.elapsed();
    println!("benchmark (cpu, host tier)");
    println!("  objects={n} bytes_per={bytes}");
    println!(
        "  create_per_op_ns={}",
        create_elapsed.as_nanos() / (n as u128)
    );
    println!("  lookup_hits={hits}");
    println!(
        "  lookup_per_op_ns={}",
        lookup_elapsed.as_nanos() / (n as u128)
    );
    println!("  hardware=CPU host (see BENCHMARKS.md for environment)");
    Ok(())
}

pub fn parse_tier(s: &str) -> Result<Tier> {
    match s {
        "host" => Ok(Tier::Host),
        "persistent" => Ok(Tier::Persistent),
        "accelerator" | "accel" => Ok(Tier::Accelerator(tensorcache::backend::BackendId::cpu(0))),
        other => Err(Error::InvalidArgument(format!("unknown tier {other}"))),
    }
}
