//! Real multi-process coordinator/node integration test.
//!
//! Spawns the `tensorcache` binary as an independent coordinator process and
//! two independent node processes, then drives them over the framed TCP
//! protocol as a client: create an object on node A, resolve and transfer it
//! to node B, migrate canonical ownership to B, and verify that a stale old
//! owner cannot retain authority after a coordinator restart.

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use tensorcache::compat::CompatKey;
use tensorcache::crc::crc32c;
use tensorcache::dtype::Dtype;
use tensorcache::geometry::{Layout, Shape};
use tensorcache::ident::Address;
use tensorcache::protocol::{read_frame, write_frame, Message};

const BIN: &str = env!("CARGO_BIN_EXE_tensorcache");

/// Kill a child on drop (best-effort; also used to "stop" the coordinator).
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn addr(p: u16) -> String {
    format!("127.0.0.1:{p}")
}

fn wait_ready(port: u16) {
    for _ in 0..200 {
        if TcpStream::connect(addr(port)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(25));
    }
    panic!("process did not become ready on port {port}");
}

fn connect(port: u16) -> TcpStream {
    let s = TcpStream::connect(addr(port)).unwrap();
    s.set_nodelay(true).ok();
    s
}

fn roundtrip(port: u16, msg: &Message) -> Message {
    let mut s = connect(port);
    write_frame(&mut s, msg).unwrap();
    let (t, payload) = read_frame(&mut s).unwrap().unwrap();
    Message::decode(t, &payload).unwrap()
}

fn spawn(mut cmd: Command) -> Child {
    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn failed")
}

fn test_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tc-dist-test-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn compat() -> CompatKey {
    CompatKey {
        dtype: Dtype::F32,
        shape: Shape::new(vec![16, 16]).unwrap(),
        layout: Layout::RowMajor,
        model: Some("model-a".into()),
        ..Default::default()
    }
}

fn payload() -> Vec<u8> {
    (0..1024).map(|i| (i % 251) as u8).collect()
}

#[test]
fn distributed_reuse_transfer_and_migration() {
    let coord_port = free_port();
    let node_a_port = free_port();
    let node_b_port = free_port();
    let dir = test_dir("dist");

    // Start the coordinator with a snapshot file.
    let snap = dir.join("coord.snap");
    let mut coord_cmd = Command::new(BIN);
    coord_cmd
        .arg("coordinator")
        .arg("--listen")
        .arg(addr(coord_port))
        .arg("--snapshot")
        .arg(snap.to_string_lossy().to_string());
    let coord = ChildGuard(spawn(coord_cmd));
    wait_ready(coord_port);

    // Start two nodes.
    let store_a = dir.join("node-a");
    let store_b = dir.join("node-b");
    let mut cmd_a = Command::new(BIN);
    cmd_a
        .arg("node")
        .arg("--id")
        .arg("n1")
        .arg("--listen")
        .arg(addr(node_a_port))
        .arg("--coordinator")
        .arg(addr(coord_port))
        .arg("--store")
        .arg(store_a.to_string_lossy().to_string());
    let node_a = ChildGuard(spawn(cmd_a));
    wait_ready(node_a_port);

    let mut cmd_b = Command::new(BIN);
    cmd_b
        .arg("node")
        .arg("--id")
        .arg("n2")
        .arg("--listen")
        .arg(addr(node_b_port))
        .arg("--coordinator")
        .arg(addr(coord_port))
        .arg("--store")
        .arg(store_b.to_string_lossy().to_string());
    let node_b = ChildGuard(spawn(cmd_b));
    wait_ready(node_b_port);

    let comp = compat();
    let comp_bytes = comp.encode();
    let data = payload();
    let oid = Address::new("ns", "k", 1).object_id().to_hex();
    let crc = crc32c(&data);

    // 1. Create the object on node A (first writer becomes owner).
    let ack = roundtrip(
        node_a_port,
        &Message::Store {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            data: data.clone(),
            crc,
            compat: comp_bytes.clone(),
            source: "n1".into(),
        },
    );
    assert!(matches!(&ack, Message::StoreAck { object_id } if object_id == &oid));

    // 2. Node B resolves and transfers the object (distributed reuse).
    let lookup = roundtrip(
        node_b_port,
        &Message::Lookup {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            compat: comp_bytes.clone(),
        },
    );
    match &lookup {
        Message::LookupResult {
            found: true, owner, ..
        } => {
            assert_eq!(owner.as_deref(), Some("n1"));
        }
        other => panic!("expected lookup hit, got {other:?}"),
    }

    // 3. Fetch from node B (now a replica) and verify the bytes.
    let fetch = roundtrip(
        node_b_port,
        &Message::Fetch {
            object_id: oid.clone(),
            compat: comp_bytes.clone(),
        },
    );
    match &fetch {
        Message::FetchReply {
            data: d, crc: c, ..
        } => {
            assert_eq!(d, &data);
            assert_eq!(*c, crc32c(d));
        }
        other => panic!("expected fetch reply, got {other:?}"),
    }

    // 4. Migrate canonical ownership from n1 to n2.
    let ack = roundtrip(
        node_a_port,
        &Message::Migrate {
            object_id: oid.clone(),
            new_owner: "n2".into(),
            new_owner_addr: addr(node_b_port),
            fence: 0,
        },
    );
    assert!(matches!(&ack, Message::MigrateAck { new_owner, .. } if new_owner == "n2"));

    // 5. Lookup now returns n2 as owner (ownership transferred).
    let lookup = roundtrip(
        node_b_port,
        &Message::Lookup {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            compat: comp_bytes.clone(),
        },
    );
    match &lookup {
        Message::LookupResult {
            found: true, owner, ..
        } => {
            assert_eq!(owner.as_deref(), Some("n2"));
        }
        other => panic!("expected migrated owner, got {other:?}"),
    }

    // 6. Stale old owner cannot retain authority: renewing with the old fence 0
    // must be rejected by the coordinator (the object fence was bumped).
    let renew = roundtrip(
        coord_port,
        &Message::LeaseRenew {
            object_id: oid.clone(),
            fence: 0,
        },
    );
    assert!(
        matches!(&renew, Message::Error { .. }),
        "stale owner must be rejected, got {renew:?}"
    );

    // 7. Coordinator restart preserves ownership from the snapshot.
    drop(coord);
    thread::sleep(Duration::from_millis(200));
    let mut coord_cmd2 = Command::new(BIN);
    coord_cmd2
        .arg("coordinator")
        .arg("--listen")
        .arg(addr(coord_port))
        .arg("--snapshot")
        .arg(snap.to_string_lossy().to_string());
    let coord2 = ChildGuard(spawn(coord_cmd2));
    wait_ready(coord_port);

    let lookup = roundtrip(
        node_b_port,
        &Message::Lookup {
            namespace: "ns".into(),
            key: "k".into(),
            generation: 1,
            compat: comp_bytes,
        },
    );
    match &lookup {
        Message::LookupResult {
            found: true, owner, ..
        } => {
            assert_eq!(owner.as_deref(), Some("n2"));
        }
        other => panic!("expected object after coordinator restart, got {other:?}"),
    }

    let _ = (coord2, node_a, node_b, dir);
}
