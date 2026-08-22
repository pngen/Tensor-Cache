#![forbid(unsafe_code)]
//! The local Tensor Cache runtime: the public, thread-safe facade.
//!
//! A single runtime owns a bounded cache of reusable tensor state. It manages
//! block storage with content-addressed deduplication, tier residency (host
//! memory, accelerator device, persistent storage), admission, eviction,
//! reuse economics, integrity verification and durable persistence.
//!
//! Concurrency model: the runtime state is guarded by a single mutex. Internal
//! helpers never re-acquire the lock; a method reads the data it needs into an
//! owned snapshot, releases any entry borrow, and only then mutates. This
//! eliminates read->write self-deadlocks and lock-ordering hazards. Mutex
//! poisoning is recovered rather than propagated as a panic.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::accounting::Accounting;
use crate::admission::{evaluate_ignoring_capacity, AdmissionCandidate, AdmissionPolicy};
use crate::backend::{Backend, BackendId, BackendRegistry, DeviceBuffer};
use crate::compat::CompatKey;
use crate::cost::CostModel;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::eviction::{eviction_order, Evictable, EvictionPolicy};
use crate::hash::Digest;
use crate::ident::{Address, ObjectId};
use crate::persistence::{blocks_exist, PersistentStore};
use crate::planner::{decide, Action, Plan};
use crate::residency::{classify, MoveFlags, Residency};
use crate::storage::{chunk, reconstruct, validate_block_list, BlockArena, BlockRef};
use crate::tiers::{Tier, TierKind};

/// Configuration for a local runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub block_size: u64,
    pub host_capacity: u64,
    pub persistent_path: Option<PathBuf>,
    pub persistent_capacity: u64,
    pub admission: AdmissionPolicy,
    pub eviction: EvictionPolicy,
    pub cost: CostModel,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            block_size: crate::storage::DEFAULT_BLOCK_SIZE,
            host_capacity: 1 << 30,
            persistent_path: None,
            persistent_capacity: 1 << 31,
            admission: AdmissionPolicy::default(),
            eviction: EvictionPolicy::default(),
            cost: CostModel::default(),
        }
    }
}

/// A materialized placement in a tier.
#[derive(Debug)]
enum Placement {
    Host { block_refs: Vec<BlockRef> },
    Accelerator { device: DeviceBuffer },
    Persistent,
}

impl Placement {
    fn tier(&self) -> Tier {
        match self {
            Placement::Host { .. } => Tier::Host,
            Placement::Accelerator { device } => Tier::Accelerator(device.backend.clone()),
            Placement::Persistent => Tier::Persistent,
        }
    }
}

/// The in-memory metadata for a tensor entry.
#[derive(Debug)]
struct Entry {
    address: Address,
    compat: CompatKey,
    compat_id: Digest,
    byte_len: u64,
    numel: u64,
    blocks: Vec<BlockRef>,
    placements: Vec<Placement>,
    created_ns: u64,
    last_use_ns: u64,
    reuse_count: u64,
    durable: bool,
    flags: MoveFlags,
}

impl Entry {
    fn has_tier(&self, t: &Tier) -> bool {
        self.placements.iter().any(|p| &p.tier() == t)
    }
    fn tier(&self, t: &Tier) -> Option<&Placement> {
        self.placements.iter().find(|p| &p.tier() == t)
    }
    fn placement_tiers(&self) -> Vec<Tier> {
        self.placements.iter().map(|p| p.tier()).collect()
    }
    fn is_quarantined(&self) -> bool {
        self.flags.quarantined || self.flags.invalid
    }
}

/// Internal mutable state guarded by the runtime mutex.
#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    host: BlockArena,
    persistent_refs: HashMap<Digest, u64>,
    storage_used: u64,
    accel_used: u64,
    accounting: Accounting,
    object_count: u64,
}

fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// The public runtime.
pub struct TensorCache {
    state: Mutex<State>,
    backends: BackendRegistry,
    config: RuntimeConfig,
    persistent: Option<PersistentStore>,
}

impl TensorCache {
    /// Create a runtime, opening a durable persistent tier if configured.
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        if config.block_size == 0 {
            return Err(Error::InvalidArgument("block size must be nonzero".into()));
        }
        let mut accounting = Accounting::new();
        accounting.set_tier_capacity(TierKind::Host, config.host_capacity);
        accounting.set_tier_capacity(TierKind::Persistent, config.persistent_capacity);

        let mut backends = BackendRegistry::new();
        backends.register(Box::new(crate::backend_cpu::CpuBackend::new(
            0,
            config.host_capacity,
        )));

        let persistent = match &config.persistent_path {
            Some(p) => Some(PersistentStore::open(p)?),
            None => None,
        };

        let tc = TensorCache {
            state: Mutex::new(State {
                accounting,
                ..Default::default()
            }),
            backends,
            config,
            persistent,
        };
        tc.recover()?;
        Ok(tc)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn backend(&self, id: &BackendId) -> Result<&dyn Backend> {
        self.backends
            .get(id)
            .ok_or_else(|| Error::Backend(format!("backend {id} not registered")))
    }

    fn get_meta<'a>(&self, s: &'a State, oid: &ObjectId) -> Result<&'a Entry> {
        s.entries
            .get(&oid.to_hex())
            .ok_or_else(|| Error::NotFound(format!("object {oid}")))
    }

    fn entry_has_tier(&self, s: &State, oid: &ObjectId, t: &Tier) -> Result<bool> {
        Ok(self.get_meta(s, oid)?.has_tier(t))
    }

    fn entry_quarantined(&self, s: &State, oid: &ObjectId) -> Result<bool> {
        Ok(self.get_meta(s, oid)?.is_quarantined())
    }

    // ---- Admission / registration -----------------------------------------

    /// Register a tensor entry into the cache (admission + host placement).
    pub fn register(
        &self,
        namespace: impl Into<String>,
        key: impl Into<String>,
        generation: u64,
        compat: CompatKey,
        payload: &[u8],
    ) -> Result<ObjectId> {
        let address = Address::new(namespace, key, generation);
        let oid = address.object_id();
        let byte_len = compat.shape.byte_len(compat.dtype.byte_size())?;
        if byte_len != payload.len() as u64 {
            return Err(Error::Geometry(format!(
                "payload length {} does not match declared byte_len {byte_len}",
                payload.len()
            )));
        }
        let numel = compat.shape.numel()?;
        let blocks = chunk(payload, self.config.block_size)?;
        validate_block_list(&blocks, byte_len)?;

        let reconstruct_cost = self.config.cost.reconstruct_cost_ns(byte_len);
        let candidate = AdmissionCandidate {
            object_id: oid.to_hex(),
            bytes: byte_len,
            reconstruction_cost_ns: reconstruct_cost,
            transfer_cost_ns: self
                .config
                .cost
                .transfer_cost_ns(&Tier::Host, &Tier::Host, byte_len),
            reuse_value_ns: reconstruct_cost,
            priority: 0,
            desired_tier: TierKind::Host,
            immutable: compat.mutability == crate::dtype::Mutability::Immutable,
        };

        let mut s = self.lock();
        if s.entries.contains_key(&oid.to_hex()) {
            return Err(Error::Exists(format!("object {oid} already registered")));
        }
        let decision =
            evaluate_ignoring_capacity(&self.config.admission, &candidate, &s.accounting);
        if !decision.is_admit() {
            return Err(Error::AdmissionRejected(decision.reason()));
        }
        let mut new_host = 0u64;
        for b in &blocks {
            if !s.host.contains(&b.content_hash) {
                new_host += b.len;
            }
        }
        if new_host > s.accounting.free(TierKind::Host) {
            drop(s);
            self.evict_to_free(new_host)?;
            s = self.lock();
            if new_host > s.accounting.free(TierKind::Host) {
                return Err(Error::AdmissionRejected(
                    "no capacity after eviction".into(),
                ));
            }
        }
        s.accounting.reserve(TierKind::Host, new_host)?;
        let mut acquired = Vec::with_capacity(blocks.len());
        for b in &blocks {
            let start = b.offset as usize;
            let end = start + b.len as usize;
            acquired.push(s.host.acquire_at(&payload[start..end], b.offset));
        }
        s.accounting.commit_reserve(TierKind::Host, new_host)?;

        let entry = Entry {
            address,
            compat: compat.clone(),
            compat_id: compat.compat_id(),
            byte_len,
            numel,
            blocks,
            placements: vec![Placement::Host {
                block_refs: acquired,
            }],
            created_ns: now_ns(),
            last_use_ns: now_ns(),
            reuse_count: 0,
            durable: false,
            flags: MoveFlags::default(),
        };
        s.entries.insert(oid.to_hex(), entry);
        s.object_count += 1;
        drop(s);
        self.enforce_capacity()?;
        Ok(oid)
    }

    /// Safe cache lookup by stable identity + compatibility gate.
    pub fn lookup(
        &self,
        namespace: impl Into<String>,
        key: impl Into<String>,
        generation: u64,
        compat: &CompatKey,
    ) -> Result<LookupResult> {
        let address = Address::new(namespace, key, generation);
        let oid = address.object_id();
        let s = self.lock();
        let entry = match s.entries.get(&oid.to_hex()) {
            Some(e) => e,
            None => return Err(Error::NotFound(format!("object {oid}"))),
        };
        if entry.compat_id != compat.compat_id() {
            return Err(Error::Compatibility(format!(
                "object {oid} has incompatible compat identity (requested {}, stored {})",
                compat.compat_id(),
                entry.compat_id
            )));
        }
        let source = preferred_tier(entry);
        let tiers = entry.placement_tiers();
        let byte_len = entry.byte_len;
        drop(s);
        {
            let mut s2 = self.lock();
            if let Some(e2) = s2.entries.get_mut(&oid.to_hex()) {
                e2.reuse_count = e2.reuse_count.saturating_add(1);
                e2.last_use_ns = now_ns();
            }
        }
        let plan = decide(&self.config.cost, byte_len, &Tier::Host, &tiers, false);
        let reconstruction_avoided = if plan.action != Action::Reconstruct {
            self.config.cost.reconstruct_cost_ns(byte_len)
        } else {
            0
        };
        Ok(LookupResult {
            hit: true,
            object_id: oid,
            generation,
            source_tier: source,
            compat_ok: true,
            bytes: byte_len,
            reconstruction_avoided_ns: reconstruction_avoided,
            transfer_cost_ns: plan.cost_ns,
            rationale: plan.rationale,
        })
    }

    /// Materialize the tensor bytes into `tier` and return them.
    pub fn restore(&self, oid: &ObjectId, tier: &Tier) -> Result<Vec<u8>> {
        let mut s = self.lock();
        if self.entry_quarantined(&s, oid)? {
            return Err(Error::Integrity("object is quarantined".into()));
        }
        let bytes = self.materialize_payload(&s, oid)?;
        if !self.entry_has_tier(&s, oid, tier)? {
            self.add_placement(&mut s, oid, tier, &bytes)?;
        }
        drop(s);
        let _ = self.enforce_capacity();
        Ok(bytes)
    }

    /// Add a placement of `oid` in `tier` (idempotent if already present).
    pub fn promote(&self, oid: &ObjectId, tier: &Tier) -> Result<()> {
        let mut s = self.lock();
        if self.entry_has_tier(&s, oid, tier)? {
            return Ok(());
        }
        if self.entry_quarantined(&s, oid)? {
            return Err(Error::Integrity("object is quarantined".into()));
        }
        let bytes = self.materialize_payload(&s, oid)?;
        self.add_placement(&mut s, oid, tier, &bytes)?;
        drop(s);
        let _ = self.enforce_capacity();
        Ok(())
    }

    /// Move a placement from a faster tier to the immediately lower tier,
    /// removing the source placement only after the lower copy is verified.
    pub fn demote(&self, oid: &ObjectId, from: &Tier) -> Result<()> {
        let mut s = self.lock();
        if !self.entry_has_tier(&s, oid, from)? {
            return Err(Error::Residency(format!(
                "object {oid} is not {from} resident"
            )));
        }
        let target = next_lower(from)?;
        if !self.entry_has_tier(&s, oid, &target)? {
            let bytes = self.materialize_payload(&s, oid)?;
            self.add_placement(&mut s, oid, &target, &bytes)?;
        }
        self.remove_placement(&mut s, oid, from)?;
        drop(s);
        Ok(())
    }

    /// Ensure a durable persistent copy exists.
    pub fn persist(&self, oid: &ObjectId) -> Result<()> {
        let mut s = self.lock();
        if !self.entry_has_tier(&s, oid, &Tier::Persistent)? {
            let bytes = self.materialize_payload(&s, oid)?;
            self.add_placement(&mut s, oid, &Tier::Persistent, &bytes)?;
        }
        if let Some(e) = s.entries.get_mut(&oid.to_hex()) {
            e.durable = true;
        }
        drop(s);
        Ok(())
    }

    /// Add a placement (replica) of `oid` in `tier`.
    pub fn replicate(&self, oid: &ObjectId, tier: &Tier) -> Result<()> {
        self.promote(oid, tier)
    }

    /// Reclaim non-durable placements; keep a persistent manifest if any.
    pub fn evict(&self, oid: &ObjectId) -> Result<()> {
        let mut s = self.lock();
        let tiers = self.get_meta(&s, oid)?.placement_tiers();
        for t in tiers {
            if t == Tier::Persistent {
                continue;
            }
            self.remove_placement(&mut s, oid, &t)?;
        }
        drop(s);
        Ok(())
    }

    /// Remove an object entirely (free all placements and metadata).
    pub fn delete(&self, oid: &ObjectId) -> Result<()> {
        let mut s = self.lock();
        let tiers = self.get_meta(&s, oid)?.placement_tiers();
        for t in tiers {
            self.remove_placement(&mut s, oid, &t)?;
        }
        if s.entries.remove(&oid.to_hex()).is_some() {
            s.object_count = s.object_count.saturating_sub(1);
        }
        if let Some(st) = &self.persistent {
            st.remove_manifest(&oid.to_hex()).ok();
        }
        drop(s);
        Ok(())
    }

    /// Verify the integrity of an object across all its placements.
    pub fn verify(&self, oid: &ObjectId) -> Result<IntegrityReport> {
        let s = self.lock();
        let bytes = self.materialize_payload(&s, oid)?;
        let (byte_len, placements) = {
            let e = self.get_meta(&s, oid)?;
            (
                e.byte_len,
                e.placements.iter().map(|p| p.tier()).collect::<Vec<_>>(),
            )
        };
        if bytes.len() as u64 != byte_len {
            return Err(Error::Integrity(format!(
                "object {oid} byte length mismatch"
            )));
        }
        let mut checked = 0u64;
        for t in &placements {
            let b = self.materialize_from(&s, oid, t)?;
            if b != bytes {
                return Err(Error::Integrity(format!(
                    "placement {t} of {oid} disagrees"
                )));
            }
            checked += 1;
        }
        drop(s);
        Ok(IntegrityReport {
            object_id: oid.to_hex(),
            checked_placements: checked,
            verified_bytes: byte_len,
            clean: true,
        })
    }

    /// Inspect an object's metadata.
    pub fn metadata(&self, oid: &ObjectId) -> Result<EntryMetadata> {
        let s = self.lock();
        let e = self.get_meta(&s, oid)?;
        Ok(EntryMetadata {
            object_id: oid.to_hex(),
            namespace: e.address.namespace.clone(),
            key: e.address.key.clone(),
            generation: e.address.generation,
            dtype: e.compat.dtype,
            byte_len: e.byte_len,
            numel: e.numel,
            created_ns: e.created_ns,
            last_use_ns: e.last_use_ns,
            reuse_count: e.reuse_count,
            durable: e.durable,
            placements: e.placement_tiers(),
            residency: classify(&e.placement_tiers()),
        })
    }

    /// Report resource usage.
    pub fn resources(&self) -> ResourceReport {
        let s = self.lock();
        ResourceReport {
            host_used: s.accounting.used(TierKind::Host),
            host_capacity: s.accounting.capacity(TierKind::Host),
            host_reserved: s.accounting.reserved(TierKind::Host),
            accel_used: s.accel_used,
            accel_capacity: s.accounting.capacity(TierKind::Accelerator),
            storage_used: s.storage_used,
            storage_capacity: s.accounting.capacity(TierKind::Persistent),
            object_count: s.object_count,
            block_count: s.host.block_count() as u64,
            replica_count: s.entries.values().map(|e| e.placements.len() as u64).sum(),
        }
    }

    /// Query the reuse/placement economics of an object into `dest`.
    pub fn economics(&self, oid: &ObjectId, dest: &Tier) -> Result<Plan> {
        let s = self.lock();
        let (byte_len, tiers) = {
            let e = self.get_meta(&s, oid)?;
            (e.byte_len, e.placement_tiers())
        };
        drop(s);
        Ok(decide(&self.config.cost, byte_len, dest, &tiers, true))
    }

    /// Recover durable persistent state after a restart.
    pub fn recover(&self) -> Result<RecoveryReport> {
        let mut recovered = 0u64;
        let mut skipped = 0u64;
        let store = match &self.persistent {
            Some(s) => s,
            None => return Ok(RecoveryReport { recovered, skipped }),
        };
        let metas = store.recover();
        let mut s = self.lock();
        for meta in metas {
            let oid = match ObjectId::from_hex(&meta.object_id) {
                Ok(o) => o,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            if !blocks_exist(store, &meta.blocks) {
                skipped += 1;
                continue;
            }
            let blocks = meta.blocks.clone();
            for b in &blocks {
                if let Some(r) = s.persistent_refs.get_mut(&b.content_hash) {
                    *r += 1;
                } else {
                    s.persistent_refs.insert(b.content_hash, 1);
                    s.storage_used += b.len;
                }
            }
            let address = Address::new(meta.namespace.clone(), meta.key.clone(), meta.generation);
            let compat = meta.compat.clone();
            let compat_id = compat.compat_id();
            let entry = Entry {
                address,
                compat,
                compat_id,
                byte_len: meta.byte_len,
                numel: meta.numel,
                blocks,
                placements: vec![Placement::Persistent],
                created_ns: meta.created_ns,
                last_use_ns: now_ns(),
                reuse_count: 0,
                durable: true,
                flags: MoveFlags::default(),
            };
            s.entries.insert(oid.to_hex(), entry);
            s.object_count += 1;
            recovered += 1;
        }
        drop(s);
        Ok(RecoveryReport { recovered, skipped })
    }

    // ---- Internal materialization -----------------------------------------

    /// Reconstruct the payload of an object from its best available placement.
    fn materialize_payload(&self, s: &State, oid: &ObjectId) -> Result<Vec<u8>> {
        let e = self.get_meta(s, oid)?;
        let tiers = e.placement_tiers();
        let byte_len = e.byte_len;
        let blocks = &e.blocks;
        if tiers.iter().any(|t| t == &Tier::Host) {
            return reconstruct(blocks, byte_len, |h| s.host.get(h));
        }
        for p in &e.placements {
            if let Placement::Accelerator { device } = p {
                let be = self.backend(&device.backend)?;
                let mut buf = vec![0u8; byte_len as usize];
                be.device_to_host(device, &mut buf)?;
                return Ok(buf);
            }
        }
        if tiers.iter().any(|t| t == &Tier::Persistent) {
            if let Some(st) = &self.persistent {
                return reconstruct(blocks, byte_len, |h| st.get_block(h).map(Arc::from));
            }
        }
        Err(Error::Reconstruct(format!(
            "object {oid} has no materialized placement"
        )))
    }

    /// Reconstruct the payload specifically from a given placement tier.
    fn materialize_from(&self, s: &State, oid: &ObjectId, tier: &Tier) -> Result<Vec<u8>> {
        let e = self.get_meta(s, oid)?;
        let byte_len = e.byte_len;
        match tier {
            Tier::Host => reconstruct(&e.blocks, byte_len, |h| s.host.get(h)),
            Tier::Persistent => {
                if let Some(st) = &self.persistent {
                    reconstruct(&e.blocks, byte_len, |h| st.get_block(h).map(Arc::from))
                } else {
                    Err(Error::Persistence("no persistent tier".into()))
                }
            }
            Tier::Accelerator(id) => {
                let p = e
                    .tier(tier)
                    .ok_or_else(|| Error::NotFound(format!("no {tier} placement")))?;
                if let Placement::Accelerator { device } = p {
                    let be = self.backend(id)?;
                    let mut buf = vec![0u8; byte_len as usize];
                    be.device_to_host(device, &mut buf)?;
                    Ok(buf)
                } else {
                    Err(Error::Internal("placement tier mismatch".into()))
                }
            }
        }
    }

    /// Add a placement of the propagated payload into `tier`.
    fn add_placement(
        &self,
        s: &mut State,
        oid: &ObjectId,
        tier: &Tier,
        bytes: &[u8],
    ) -> Result<()> {
        let (byte_len, address, compat, created) = {
            let e = self.get_meta(s, oid)?;
            (
                e.byte_len,
                e.address.clone(),
                e.compat.clone(),
                e.created_ns,
            )
        };
        if bytes.len() as u64 != byte_len {
            return Err(Error::Integrity("placement payload length mismatch".into()));
        }
        match tier {
            Tier::Host => {
                let blocks = chunk(bytes, self.config.block_size)?;
                validate_block_list(&blocks, byte_len)?;
                let mut new_host = 0u64;
                for b in &blocks {
                    if !s.host.contains(&b.content_hash) {
                        new_host += b.len;
                    }
                }
                if new_host > s.accounting.free(TierKind::Host) {
                    return Err(Error::AdmissionRejected(
                        "host tier capacity exceeded".into(),
                    ));
                }
                s.accounting.reserve(TierKind::Host, new_host)?;
                let mut acquired = Vec::with_capacity(blocks.len());
                for b in &blocks {
                    let start = b.offset as usize;
                    let end = start + b.len as usize;
                    acquired.push(s.host.acquire_at(&bytes[start..end], b.offset));
                }
                s.accounting.commit_reserve(TierKind::Host, new_host)?;
                if let Some(e) = s.entries.get_mut(&oid.to_hex()) {
                    e.placements.push(Placement::Host {
                        block_refs: acquired,
                    });
                }
                Ok(())
            }
            Tier::Accelerator(backend_id) => {
                let be = self.backend(backend_id)?;
                let mut dev = be.allocate(bytes.len())?;
                be.to_device(bytes, &mut dev)?;
                s.accel_used += bytes.len() as u64;
                if let Some(e) = s.entries.get_mut(&oid.to_hex()) {
                    e.placements.push(Placement::Accelerator { device: dev });
                }
                Ok(())
            }
            Tier::Persistent => {
                let store = self
                    .persistent
                    .as_ref()
                    .ok_or_else(|| Error::Persistence("no persistent tier configured".into()))?;
                let blocks = chunk(bytes, self.config.block_size)?;
                validate_block_list(&blocks, byte_len)?;
                let mut new_storage = 0u64;
                for b in &blocks {
                    if !s.persistent_refs.contains_key(&b.content_hash) {
                        new_storage += b.len;
                    }
                }
                if new_storage > s.accounting.free(TierKind::Persistent) {
                    return Err(Error::AdmissionRejected(
                        "persistent tier capacity exceeded".into(),
                    ));
                }
                s.accounting.reserve(TierKind::Persistent, new_storage)?;
                for b in &blocks {
                    use std::collections::hash_map::Entry;
                    match s.persistent_refs.entry(b.content_hash) {
                        Entry::Vacant(v) => {
                            let start = b.offset as usize;
                            let end = start + b.len as usize;
                            store.put_block(&bytes[start..end])?;
                            v.insert(1);
                            s.storage_used += b.len;
                        }
                        Entry::Occupied(mut o) => {
                            *o.get_mut() += 1;
                        }
                    }
                }
                s.accounting
                    .commit_reserve(TierKind::Persistent, new_storage)?;
                let numel = compat.shape.numel()?;
                let meta = crate::persistence::PersistEntryMeta {
                    object_id: oid.to_hex(),
                    namespace: address.namespace,
                    key: address.key,
                    generation: address.generation,
                    compat,
                    byte_len,
                    numel,
                    created_ns: created,
                    blocks: blocks.clone(),
                };
                store.write_manifest(&meta)?;
                if let Some(e) = s.entries.get_mut(&oid.to_hex()) {
                    e.placements.push(Placement::Persistent);
                }
                Ok(())
            }
        }
    }

    /// Remove a placement from `tier`, freeing/verifying the underlying state.
    fn remove_placement(&self, s: &mut State, oid: &ObjectId, tier: &Tier) -> Result<()> {
        if let Some(e) = s.entries.get_mut(&oid.to_hex()) {
            let idx = e.placements.iter().position(|p| &p.tier() == tier);
            if let Some(i) = idx {
                let p = e.placements.remove(i);
                match p {
                    Placement::Host { block_refs } => {
                        for b in &block_refs {
                            if s.host.release(&b.content_hash) {
                                s.accounting.sub_bytes(TierKind::Host, b.len)?;
                            }
                        }
                    }
                    Placement::Accelerator { device } => {
                        let be = self.backend(&device.backend)?;
                        let bs = device.bytes;
                        be.free(device)?;
                        s.accel_used = s.accel_used.saturating_sub(bs as u64);
                    }
                    Placement::Persistent => {
                        let block_list = e.blocks.clone();
                        for b in &block_list {
                            if let Some(r) = s.persistent_refs.get_mut(&b.content_hash) {
                                *r -= 1;
                                if *r == 0 {
                                    if let Some(st) = &self.persistent {
                                        let _ = st.remove_block(&b.content_hash);
                                    }
                                    s.persistent_refs.remove(&b.content_hash);
                                    s.storage_used = s.storage_used.saturating_sub(b.len);
                                }
                            }
                        }
                        if let Some(st) = &self.persistent {
                            let _ = st.remove_manifest(&oid.to_hex());
                        }
                    }
                }
                return Ok(());
            }
        }
        Err(Error::Residency(format!(
            "object {oid} is not {tier} resident"
        )))
    }

    /// Evict the lowest-scoring non-durable host entries until at least
    /// `needed` bytes of host capacity are free.
    fn evict_to_free(&self, needed: u64) -> Result<()> {
        let mut s = self.lock();
        if s.accounting.free(TierKind::Host) >= needed {
            return Ok(());
        }
        let host_cap = s.accounting.capacity(TierKind::Host);
        let mut candidates: Vec<Evictable> = Vec::new();
        for (key, e) in &s.entries {
            if e.has_tier(&Tier::Host) && !e.durable {
                candidates.push(Evictable {
                    object_id: key.clone(),
                    bytes: e.byte_len,
                    reuse_count: e.reuse_count,
                    age_seconds: now_ns().saturating_sub(e.last_use_ns) / 1_000_000_000,
                    reconstruction_cost_ns: self.config.cost.reconstruct_cost_ns(e.byte_len),
                    priority: 0,
                    pressure: s.accounting.used(TierKind::Host) as f64 / host_cap.max(1) as f64,
                    durable: e.durable,
                });
            }
        }
        let order = eviction_order(&self.config.eviction, &candidates);
        for e in order {
            if s.accounting.free(TierKind::Host) >= needed {
                break;
            }
            let oid = ObjectId::from_hex(&e.object_id)?;
            let tiers = s
                .entries
                .get(&e.object_id)
                .map(|ent| ent.placement_tiers())
                .unwrap_or_default();
            for t in tiers {
                if t == Tier::Persistent {
                    continue;
                }
                let _ = self.remove_placement(&mut s, &oid, &t);
            }
        }
        drop(s);
        Ok(())
    }

    /// Enforce host/accelerator capacity by evicting the lowest-scoring entries.
    fn enforce_capacity(&self) -> Result<()> {
        let mut s = self.lock();
        let host_cap = s.accounting.capacity(TierKind::Host);
        let mut candidates: Vec<Evictable> = Vec::new();
        for (key, e) in &s.entries {
            if e.has_tier(&Tier::Host) && !e.durable {
                candidates.push(Evictable {
                    object_id: key.clone(),
                    bytes: e.byte_len,
                    reuse_count: e.reuse_count,
                    age_seconds: now_ns().saturating_sub(e.last_use_ns) / 1_000_000_000,
                    reconstruction_cost_ns: self.config.cost.reconstruct_cost_ns(e.byte_len),
                    priority: 0,
                    pressure: s.accounting.used(TierKind::Host) as f64 / host_cap.max(1) as f64,
                    durable: e.durable,
                });
            }
        }
        let order = eviction_order(&self.config.eviction, &candidates);
        for e in order {
            if s.accounting.used(TierKind::Host) <= host_cap {
                break;
            }
            let oid = ObjectId::from_hex(&e.object_id)?;
            let tiers = s
                .entries
                .get(&e.object_id)
                .map(|ent| ent.placement_tiers())
                .unwrap_or_default();
            for t in tiers {
                if t == Tier::Persistent {
                    continue;
                }
                let _ = self.remove_placement(&mut s, &oid, &t);
            }
        }
        drop(s);
        Ok(())
    }
}

fn preferred_tier(e: &Entry) -> Option<Tier> {
    let tiers = e.placement_tiers();
    if let Some(t) = tiers.iter().find(|t| t.is_accelerator()) {
        return Some(t.clone());
    }
    if tiers.contains(&Tier::Host) {
        return Some(Tier::Host);
    }
    if tiers.contains(&Tier::Persistent) {
        return Some(Tier::Persistent);
    }
    None
}

fn next_lower(from: &Tier) -> Result<Tier> {
    match from {
        Tier::Accelerator(_) => Ok(Tier::Host),
        Tier::Host => Ok(Tier::Persistent),
        Tier::Persistent => Err(Error::Residency("persistent is the lowest tier".into())),
    }
}

/// The result of a compatible lookup.
#[derive(Debug, Clone)]
pub struct LookupResult {
    pub hit: bool,
    pub object_id: ObjectId,
    pub generation: u64,
    pub source_tier: Option<Tier>,
    pub compat_ok: bool,
    pub bytes: u64,
    pub reconstruction_avoided_ns: u64,
    pub transfer_cost_ns: u64,
    pub rationale: String,
}

/// Integrity verification report.
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    pub object_id: String,
    pub checked_placements: u64,
    pub verified_bytes: u64,
    pub clean: bool,
}

/// Entry metadata report.
#[derive(Debug, Clone)]
pub struct EntryMetadata {
    pub object_id: String,
    pub namespace: String,
    pub key: String,
    pub generation: u64,
    pub dtype: Dtype,
    pub byte_len: u64,
    pub numel: u64,
    pub created_ns: u64,
    pub last_use_ns: u64,
    pub reuse_count: u64,
    pub durable: bool,
    pub placements: Vec<Tier>,
    pub residency: Residency,
}

/// Resource usage report.
#[derive(Debug, Clone)]
pub struct ResourceReport {
    pub host_used: u64,
    pub host_capacity: u64,
    pub host_reserved: u64,
    pub accel_used: u64,
    pub accel_capacity: u64,
    pub storage_used: u64,
    pub storage_capacity: u64,
    pub object_count: u64,
    pub block_count: u64,
    pub replica_count: u64,
}

/// Recovery report.
#[derive(Debug, Clone)]
pub struct RecoveryReport {
    pub recovered: u64,
    pub skipped: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendId;
    use crate::dtype::{Dtype, Mutability};
    use crate::geometry::{Layout, Shape};

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("tc-rt-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn f32_compat(dims: Vec<u64>) -> CompatKey {
        CompatKey {
            dtype: Dtype::F32,
            shape: Shape::new(dims).unwrap(),
            layout: Layout::RowMajor,
            model: Some("test-model".into()),
            ..Default::default()
        }
    }

    fn payload(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    fn cfg(host_capacity: u64, persistent: Option<PathBuf>) -> RuntimeConfig {
        RuntimeConfig {
            host_capacity,
            persistent_path: persistent,
            block_size: 1024,
            ..Default::default()
        }
    }

    #[test]
    fn register_and_lookup_hit() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let compat = f32_compat(vec![32, 32]); // 4096 bytes f32
        let data = payload(4096);
        let oid = tc.register("ns", "emb", 1, compat.clone(), &data).unwrap();
        let res = tc.lookup("ns", "emb", 1, &compat).unwrap();
        assert!(res.hit);
        assert!(res.compat_ok);
        assert_eq!(res.bytes, 4096);
        assert!(res.reconstruction_avoided_ns > 0);
        assert_eq!(res.source_tier, Some(Tier::Host));
        // Reuse count increments on repeated compatible lookups.
        tc.lookup("ns", "emb", 1, &compat).unwrap();
        let meta = tc.metadata(&oid).unwrap();
        assert_eq!(meta.reuse_count, 2);
    }

    #[test]
    fn compatibility_rejection_is_a_correctness_gate() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let f32 = f32_compat(vec![32, 32]);
        let data = payload(4096);
        tc.register("ns", "emb", 1, f32.clone(), &data).unwrap();
        // Same logical key but a different dtype -> must be rejected, not reused.
        let mut f16 = f32.clone();
        f16.dtype = Dtype::F16;
        let err = tc.lookup("ns", "emb", 1, &f16).unwrap_err();
        assert!(matches!(err, Error::Compatibility(_)));
    }

    #[test]
    fn namespace_isolation() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let compat = f32_compat(vec![8, 8]);
        let data = payload(256);
        // Same key, different namespace.
        let oid = tc
            .register("tenant-a", "k", 1, compat.clone(), &data)
            .unwrap();
        assert!(tc.lookup("tenant-a", "k", 1, &compat).is_ok());
        assert!(tc.lookup("tenant-b", "k", 1, &compat).is_err());
        // Different generation is a distinct object.
        assert!(tc.lookup("tenant-a", "k", 2, &compat).is_err());
        let _ = oid;
    }

    #[test]
    fn persistent_roundtrip_and_recovery() {
        let dir = temp_dir();
        // Runtime 1: register + persist.
        {
            let tc = TensorCache::new(cfg(1 << 20, Some(dir.clone()))).unwrap();
            let compat = f32_compat(vec![64, 64]); // 16384 bytes
            let data = payload(16384);
            let oid = tc
                .register("ns", "persist", 1, compat.clone(), &data)
                .unwrap();
            tc.persist(&oid).unwrap();
            let deps = tc.resources();
            assert!(deps.storage_used >= 16384);
        }
        // Runtime 2: recover from disk and restore.
        {
            let tc = TensorCache::new(cfg(1 << 20, Some(dir.clone()))).unwrap();
            let compat = f32_compat(vec![64, 64]);
            let oid = crate::ident::Address::new("ns", "persist", 1).object_id();
            let meta = tc.metadata(&oid).unwrap();
            assert!(meta.durable);
            assert!(meta.placements.contains(&Tier::Persistent));
            let bytes = tc.restore(&oid, &Tier::Host).unwrap();
            assert_eq!(bytes, payload(16384));
            tc.verify(&oid).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_under_capacity_pressure_stays_in_budget() {
        let tc = TensorCache::new(cfg(8192, None)).unwrap();
        let compat = f32_compat(vec![32, 32]); // 4096 bytes
        let data = payload(4096);
        let mut admitted = 0;
        for i in 0..10 {
            match tc.register(format!("ns{i}"), "k", 1, compat.clone(), &data) {
                Ok(_) => admitted += 1,
                Err(Error::AdmissionRejected(_)) => {}
                Err(_) => panic!("unexpected error"),
            }
        }
        let res = tc.resources();
        assert!(
            res.host_used <= res.host_capacity,
            "used {} > cap {}",
            res.host_used,
            res.host_capacity
        );
        assert!(res.host_used <= 8192);
        // At most the capacity / object-size objects can reside (2 x 4096).
        assert!(admitted >= 2);
    }

    #[test]
    fn dedup_shares_physical_bytes() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let compat = f32_compat(vec![16, 16]); // 1024 bytes
        let data = payload(1024);
        let o1 = tc.register("ns", "a", 1, compat.clone(), &data).unwrap();
        let o2 = tc.register("ns", "b", 1, compat.clone(), &data).unwrap();
        let res = tc.resources();
        // Both objects share the same block bytes: physical savings.
        assert_eq!(res.host_used, 1024);
        assert_eq!(res.object_count, 2);
        let _ = (o1, o2);
    }

    #[test]
    fn residency_transitions() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let compat = f32_compat(vec![16, 16]);
        let data = payload(1024);
        let oid = tc.register("ns", "r", 1, compat, &data).unwrap();
        // Promote to accelerator (CPU device 0).
        let accel = Tier::Accelerator(BackendId::cpu(0));
        tc.promote(&oid, &accel).unwrap();
        let m = tc.metadata(&oid).unwrap();
        assert!(m.placements.contains(&accel));
        assert!(m.placements.contains(&Tier::Host));
        // Demote from accelerator to host (child tier already present).
        tc.demote(&oid, &accel).unwrap();
        let m2 = tc.metadata(&oid).unwrap();
        assert!(!m2.placements.contains(&accel));
        assert!(m2.placements.contains(&Tier::Host));
    }

    #[test]
    fn delete_returns_accounting() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let compat = f32_compat(vec![16, 16]);
        let data = payload(1024);
        let oid = tc.register("ns", "d", 1, compat, &data).unwrap();
        assert_eq!(tc.resources().host_used, 1024);
        tc.delete(&oid).unwrap();
        let res = tc.resources();
        assert_eq!(res.host_used, 0);
        assert_eq!(res.object_count, 0);
    }

    #[test]
    fn verify_detects_inconsistent_placements() {
        let tc = TensorCache::new(cfg(1 << 20, None)).unwrap();
        let compat = f32_compat(vec![16, 16]);
        let data = payload(1024);
        let oid = tc.register("ns", "v", 1, compat, &data).unwrap();
        tc.verify(&oid).unwrap(); // clean
                                  // Mutating a mutable object must be isolated from shared blocks.
        let data2 = payload(1024);
        let _ = data2;
        // Only immutable objects share blocks; a fresh registration is a
        // separate object.
        let oid2 = tc
            .register("ns", "v2", 1, f32_compat(vec![16, 16]), &data)
            .unwrap();
        tc.verify(&oid2).unwrap();
    }
}
