//! Coordinator and node server processes, plus the client-driven migrate
//! command. These run as independent OS processes over TCP.

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tensorcache::coordinator::Coordinator;
use tensorcache::error::{Error, Result};
use tensorcache::node::Node;
use tensorcache::protocol::{read_frame, write_frame, Message};
use tensorcache::runtime::RuntimeConfig;
use tensorcache::runtime::TensorCache;

use crate::args;

/// Client-driven migration: send a Migrate request to a node and await ack.
pub fn cmd_migrate(flags: &std::collections::HashMap<String, String>) -> Result<()> {
    let node_addr = args::req(flags, "node-addr")?;
    let object_id = args::req(flags, "object")?;
    let new_owner = args::req(flags, "to")?;
    let to_addr = args::req(flags, "to-addr")?;
    let fence = args::num(flags, "fence", 0)?;
    let mut stream = TcpStream::connect(node_addr)?;
    write_frame(
        &mut stream,
        &Message::Migrate {
            object_id: object_id.to_string(),
            new_owner: new_owner.to_string(),
            new_owner_addr: to_addr.to_string(),
            fence,
        },
    )?;
    let (t, payload) =
        read_frame(&mut stream)?.ok_or_else(|| Error::Protocol("node closed".into()))?;
    match Message::decode(t, &payload)? {
        Message::MigrateAck {
            object_id,
            new_owner,
            ..
        } => println!("migrated {object_id} to {new_owner}"),
        Message::Error { message, .. } => return Err(Error::Protocol(message)),
        other => {
            return Err(Error::Protocol(format!(
                "unexpected reply {:?}",
                other.msg_type()
            )))
        }
    }
    Ok(())
}

fn open_store(dir: &str, capacity: u64) -> Result<TensorCache> {
    let config = RuntimeConfig {
        host_capacity: capacity.max(1 << 20),
        persistent_path: Some(PathBuf::from(dir)),
        ..Default::default()
    };
    #[cfg(feature = "cuda")]
    {
        let cuda = tensorcache_cuda::CudaBackend::new(0, capacity.max(1 << 20))?;
        TensorCache::with_backends(config, vec![Box::new(cuda)])
    }
    #[cfg(not(feature = "cuda"))]
    {
        TensorCache::new(config)
    }
}

/// Run the coordinator server process.
pub fn run_coordinator(flags: &std::collections::HashMap<String, String>) -> Result<()> {
    let listen = args::req(flags, "listen")?;
    let lease_ns = args::num(flags, "lease-ns", 20_000_000_000)?;
    let mut coord = Coordinator::new(lease_ns);
    let snap = args::opt(flags, "snapshot").map(PathBuf::from);
    if let Some(p) = &snap {
        coord.load_state(p)?;
    }
    let coord = Arc::new(Mutex::new(coord));
    let listener = TcpListener::bind(listen)?;
    eprintln!("tensorcache coordinator listening on {listen} (lease {lease_ns}ns)");
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let coord = Arc::clone(&coord);
        let snap = snap.clone();
        std::thread::spawn(move || {
            let _ = serve_coordinator_connection(stream, &coord, snap.as_deref());
        });
    }
    Ok(())
}

fn serve_coordinator_connection(
    mut stream: TcpStream,
    coord: &Mutex<Coordinator>,
    snap: Option<&Path>,
) -> Result<()> {
    while let Ok(Some((t, payload))) = read_frame(&mut stream) {
        let msg = match Message::decode(t, &payload) {
            Ok(m) => m,
            Err(e) => {
                let _ = write_frame(
                    &mut stream,
                    &Message::Error {
                        code: "protocol".into(),
                        message: e.to_string(),
                    },
                );
                break;
            }
        };
        let responses = {
            let mut g = coord.lock().unwrap_or_else(|p| p.into_inner());
            let r = match g.handle(&msg) {
                Ok(r) => r,
                Err(e) => vec![Message::Error {
                    code: e.kind().into(),
                    message: e.to_string(),
                }],
            };
            if let Some(p) = snap {
                let _ = g.save_state(p);
            }
            r
        };
        for r in &responses {
            write_frame(&mut stream, r)?;
        }
    }
    Ok(())
}

/// Run a storage node server process.
pub fn run_node(flags: &std::collections::HashMap<String, String>) -> Result<()> {
    let node_id = args::req(flags, "id")?;
    let listen = args::req(flags, "listen")?;
    let coord_addr = args::req(flags, "coordinator")?;
    let store = args::req(flags, "store")?;
    let lease_ns = args::num(flags, "lease-ns", 20_000_000_000)?;
    let capacity = args::num(flags, "capacity", 1 << 30)?;
    let tc = open_store(store, capacity)?;
    let node = Arc::new(Node::new(
        node_id.to_string(),
        listen.to_string(),
        coord_addr.to_string(),
        tc,
        lease_ns,
    ));
    match node.register() {
        Ok(Message::Hello {
            epoch,
            boot_id,
            node_id,
            ..
        }) => {
            eprintln!("tensorcache node {node_id} registered (epoch {epoch}, boot {boot_id})");
        }
        Ok(other) => {
            return Err(Error::Protocol(format!(
                "unexpected register reply {:?}",
                other.msg_type()
            )))
        }
        Err(e) => return Err(Error::Io(format!("coordinator handshake failed: {e}"))),
    }
    let listener = TcpListener::bind(listen)?;
    eprintln!("tensorcache node {node_id} serving on {listen}");
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let node = Arc::clone(&node);
        std::thread::spawn(move || {
            let _ = serve_node_connection(stream, &node);
        });
    }
    Ok(())
}

fn serve_node_connection(mut stream: TcpStream, node: &Node) -> Result<()> {
    while let Ok(Some((t, payload))) = read_frame(&mut stream) {
        let msg = match Message::decode(t, &payload) {
            Ok(m) => m,
            Err(e) => {
                let _ = write_frame(
                    &mut stream,
                    &Message::Error {
                        code: "protocol".into(),
                        message: e.to_string(),
                    },
                );
                break;
            }
        };
        let responses = match node.handle_peer(&msg) {
            Ok(r) => r,
            Err(e) => vec![Message::Error {
                code: e.kind().into(),
                message: e.to_string(),
            }],
        };
        for r in &responses {
            write_frame(&mut stream, r)?;
        }
    }
    Ok(())
}
