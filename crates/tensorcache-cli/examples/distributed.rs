//! Examples 5 & 6: distributed reuse and migration.
//!
//! Spawns a real coordinator process and two node processes (the tensorcache
//! binary) and drives them over the framed TCP protocol: node A owns a tensor,
//! node B resolves and transfers it (distributed reuse), then ownership is
//! migrated to B and the stale old owner cannot retain authority.
//!
//! Prerequisite: build the CLI first (`cargo build --package tensorcache-cli`).

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use tensorcache::compat::CompatKey;
use tensorcache::crc::crc32c;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::ident::Address;
use tensorcache::protocol::{read_frame, write_frame, Message};

struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn find_bin() -> Option<PathBuf> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    // manifest = <repo>/crates/tensorcache-cli; workspace target is <repo>/target.
    let repo = Path::new(manifest).join("../..");
    let exe = if cfg!(windows) {
        "tensorcache.exe"
    } else {
        "tensorcache"
    };
    let mut candidates = Vec::new();
    for d in [
        repo.join("target").join("debug"),
        repo.join("target").join("release"),
    ] {
        candidates.push(d.join(exe));
    }
    candidates.into_iter().find(|p| p.exists())
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
fn addr(p: u16) -> String {
    format!("127.0.0.1:{p}")
}
fn wait_ready(p: u16) {
    for _ in 0..200 {
        if TcpStream::connect(addr(p)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(25));
    }
    panic!("process did not become ready");
}
fn roundtrip(p: u16, msg: &Message) -> Message {
    let mut s = TcpStream::connect(addr(p)).unwrap();
    s.set_nodelay(true).ok();
    write_frame(&mut s, msg).unwrap();
    let (t, payload) = read_frame(&mut s).unwrap().unwrap();
    Message::decode(t, &payload).unwrap()
}
fn spawn(bin: &Path, args: &[&str]) -> Child {
    Command::new(bin)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}
fn store_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tc-ex-dist-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bin = match find_bin() {
        Some(b) => b,
        None => {
            println!("tensorcache CLI binary not found; run 'cargo build --package tensorcache-cli' first.");
            println!("(Real multi-process reuse + migration is also validated in crates/tensorcache-cli/tests/distributed.rs)");
            return Ok(());
        }
    };

    let coord_port = free_port();
    let a_port = free_port();
    let b_port = free_port();
    let dir = store_dir("two");

    let coord = Guard(spawn(&bin, &["coordinator", "--listen", &addr(coord_port)]));
    wait_ready(coord_port);
    let node_a = Guard(spawn(
        &bin,
        &[
            "node",
            "--id",
            "n1",
            "--listen",
            &addr(a_port),
            "--coordinator",
            &addr(coord_port),
            "--store",
            &dir.join("a").to_string_lossy(),
        ],
    ));
    wait_ready(a_port);
    let node_b = Guard(spawn(
        &bin,
        &[
            "node",
            "--id",
            "n2",
            "--listen",
            &addr(b_port),
            "--coordinator",
            &addr(coord_port),
            "--store",
            &dir.join("b").to_string_lossy(),
        ],
    ));
    wait_ready(b_port);

    let compat = CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![16, 16]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("model-a".into()),
        ..Default::default()
    };
    let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let crc = crc32c(&data);
    let oid = Address::new("ns", "k", 1).object_id().to_hex();

    // Example 5: distributed reuse (node A owns, node B transfers + replica).
    roundtrip(
        a_port,
        &Message::Store {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            data: data.clone(),
            crc,
            compat: compat.encode(),
            source: "n1".into(),
        },
    );
    println!("[reuse] node A owns {oid}");
    let lk = roundtrip(
        b_port,
        &Message::Lookup {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            compat: compat.encode(),
        },
    );
    if let Message::LookupResult {
        found: true,
        owner,
        owner_addr,
        ..
    } = &lk
    {
        println!("[reuse] node B resolved: owner={owner:?} peer={owner_addr:?}");
    }
    let fetch = roundtrip(
        b_port,
        &Message::Fetch {
            object_id: oid.clone(),
            compat: compat.encode(),
        },
    );
    if let Message::FetchReply { data: d, .. } = &fetch {
        assert_eq!(d, &data);
        println!(
            "[reuse] node B served {} bytes identical (real TCP transfer)",
            d.len()
        );
    }

    // Example 6: migration + stale owner authority.
    roundtrip(
        a_port,
        &Message::Migrate {
            object_id: oid.clone(),
            new_owner: "n2".into(),
            new_owner_addr: addr(b_port),
            fence: 0,
        },
    );
    println!("[migration] migrated {oid} from n1 to n2 (fence bumped)");
    let lk2 = roundtrip(
        b_port,
        &Message::Lookup {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            compat: compat.encode(),
        },
    );
    if let Message::LookupResult {
        found: true, owner, ..
    } = &lk2
    {
        println!("[migration] owner after migration = {owner:?}");
    }
    let renew = roundtrip(
        coord_port,
        &Message::LeaseRenew {
            object_id: oid.clone(),
            fence: 0,
        },
    );
    if let Message::Error { message, .. } = &renew {
        println!("[migration] stale old owner rejected: {message}");
    } else {
        panic!("stale authority should be rejected, got {renew:?}");
    }

    let _ = (coord, node_a, node_b, dir);
    Ok(())
}
