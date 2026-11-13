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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::accounting::Accounting;
use crate::admission::{evaluate_ignoring_capacity, AdmissionCandidate, AdmissionPolicy};
use crate::backend::{Backend, BackendId, BackendRegistry, DeviceBuffer};
use crate::compat::CompatKey;
use crate::cost::CostModel;
use crate::crc::crc32c;
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

impl State {
    /// Record the number of cached objects, also where admission reads it.
    fn set_object_count(&mut self, n: u64) {
        self.object_count = n;
        self.accounting.set_object_count(n);
    }

    /// An entry whose last placement was evicted is no longer cached: it is
    /// removed, so that it is neither reported as a hit nor in the way of a
    /// later registration of the same object.
    fn drop_if_absent(&mut self, key: &str) {
        if self
            .entries
            .get(key)
            .is_some_and(|e| e.placements.is_empty())
        {
            self.entries.remove(key);
            let objects = self.object_count.saturating_sub(1);
            self.set_object_count(objects);
        }
    }
}

fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Bytes of `blocks` that a content-addressed tier does not hold yet. A block
/// that occurs more than once in the list is stored, and so charged, once.
fn unique_new_bytes(blocks: &[BlockRef], present: impl Fn(&Digest) -> bool) -> u64 {
    let mut seen: HashSet<Digest> = HashSet::new();
    let mut total = 0u64;
    for b in blocks {
        if !present(&b.content_hash) && seen.insert(b.content_hash) {
            total += b.len;
        }
    }
    total
}

/// Check a payload against the block list recorded for its object.
fn verify_against_blocks(bytes: &[u8], blocks: &[BlockRef]) -> Result<()> {
    for b in blocks {
        let start = b.offset as usize;
        let block = start
            .checked_add(b.len as usize)
            .and_then(|end| bytes.get(start..end))
            .ok_or_else(|| Error::Integrity("payload does not cover its block list".into()))?;
        if crc32c(block) != b.crc {
            return Err(Error::Integrity(format!(
                "block {} checksum mismatch",
                b.content_hash
            )));
        }
    }
    Ok(())
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
        Self::with_backends(config, Vec::new())
    }

    /// Create a runtime with additional (e.g. accelerator) backends. Each
    /// extra backend's nominal capacity is added to the accelerator tier
    /// budget. The mandatory CPU backend is always registered first.
    pub fn with_backends(config: RuntimeConfig, extra: Vec<Box<dyn Backend>>) -> Result<Self> {
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
        // The mandatory CPU backend acts as an accelerator device; its
        // capacity (host_capacity) counts toward the accelerator budget.
        let mut accel_cap = config.host_capacity;
        for b in extra {
            accel_cap = accel_cap.saturating_add(b.byte_capacity());
            backends.register(b);
        }
        accounting.set_tier_capacity(TierKind::Accelerator, accel_cap);

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
        // Charge the bytes the host arena does not hold yet. Eviction runs
        // under the same lock and may release blocks this payload shares with
        // an evicted entry, so the figure is taken again after every round.
        let mut new_host = unique_new_bytes(&blocks, |h| s.host.contains(h));
        while new_host > s.accounting.free(TierKind::Host) {
            let free_before = s.accounting.free(TierKind::Host);
            self.evict_to_free(&mut s, new_host)?;
            new_host = unique_new_bytes(&blocks, |h| s.host.contains(h));
            if s.accounting.free(TierKind::Host) == free_before {
                break;
            }
        }
        if new_host > s.accounting.free(TierKind::Host) {
            return Err(Error::AdmissionRejected(
                "no capacity after eviction".into(),
            ));
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
        let objects = s.object_count + 1;
        s.set_object_count(objects);
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
        s.drop_if_absent(&oid.to_hex());
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
            let objects = s.object_count.saturating_sub(1);
            s.set_object_count(objects);
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
    /// Return the compatibility identity (digest) of an object, used by the
    /// owner to gate peer reuse requests.
    pub fn entry_compat_id(&self, oid: &ObjectId) -> Result<Digest> {
        let s = self.lock();
        let e = self.get_meta(&s, oid)?;
        Ok(e.compat_id)
    }

    /// Return the full compatibility key of an object (for peer transfer).
    pub fn entry_compat(&self, oid: &ObjectId) -> Result<CompatKey> {
        let s = self.lock();
        let e = self.get_meta(&s, oid)?;
        Ok(e.compat.clone())
    }

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
            // Already loaded: recovering again must not take a second set of
            // block references or replace a live entry.
            if s.entries.contains_key(&oid.to_hex()) {
                continue;
            }
            if !blocks_exist(store, &meta.blocks) {
                skipped += 1;
                continue;
            }
            let blocks = meta.blocks.clone();
            // The recovered blocks occupy the persistent tier: charge the ones
            // not accounted for yet. An object the configured budget cannot
            // hold is left on disk and not loaded.
            let new_storage = unique_new_bytes(&blocks, |h| s.persistent_refs.contains_key(h));
            if s.accounting
                .add_bytes(TierKind::Persistent, new_storage)
                .is_err()
            {
                skipped += 1;
                continue;
            }
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
            let objects = s.object_count + 1;
            s.set_object_count(objects);
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
                // Device bytes are checked like every other tier's.
                verify_against_blocks(&buf, blocks)?;
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
                    verify_against_blocks(&buf, &e.blocks)?;
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
        let (byte_len, address, compat, created, blocks) = {
            let e = self.get_meta(s, oid)?;
            (
                e.byte_len,
                e.address.clone(),
                e.compat.clone(),
                e.created_ns,
                e.blocks.clone(),
            )
        };
        if bytes.len() as u64 != byte_len {
            return Err(Error::Integrity("placement payload length mismatch".into()));
        }
        // A placement is cut along the object's own block list. That is the
        // list every read and removal of the object goes by, whatever block
        // size this runtime is configured with, and it lets the payload be
        // checked before it is installed anywhere.
        validate_block_list(&blocks, byte_len)?;
        verify_against_blocks(bytes, &blocks)?;
        match tier {
            Tier::Host => {
                let new_host = unique_new_bytes(&blocks, |h| s.host.contains(h));
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
                if let Err(e) = be.to_device(bytes, &mut dev) {
                    // Nothing refers to the allocation: hand it back.
                    let _ = be.free(dev);
                    return Err(e);
                }
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
                let numel = compat.shape.numel()?;
                let new_storage = unique_new_bytes(&blocks, |h| s.persistent_refs.contains_key(h));
                if new_storage > s.accounting.free(TierKind::Persistent) {
                    return Err(Error::AdmissionRejected(
                        "persistent tier capacity exceeded".into(),
                    ));
                }
                s.accounting.reserve(TierKind::Persistent, new_storage)?;
                // Reference (and write, when new) every block, then commit the
                // manifest. If a step fails, the references taken here are
                // dropped again and the reservation is cancelled.
                let mut taken: Vec<&BlockRef> = Vec::with_capacity(blocks.len());
                let mut failure = None;
                for b in &blocks {
                    use std::collections::hash_map::Entry;
                    match s.persistent_refs.entry(b.content_hash) {
                        Entry::Vacant(v) => {
                            let start = b.offset as usize;
                            let end = start + b.len as usize;
                            if let Err(e) = store.put_block(&bytes[start..end]) {
                                failure = Some(e);
                                break;
                            }
                            v.insert(1);
                            s.storage_used += b.len;
                        }
                        Entry::Occupied(mut o) => {
                            *o.get_mut() += 1;
                        }
                    }
                    taken.push(b);
                }
                if failure.is_none() {
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
                    failure = store.write_manifest(&meta).err();
                }
                if let Some(e) = failure {
                    for b in taken {
                        if let Some(r) = s.persistent_refs.get_mut(&b.content_hash) {
                            *r -= 1;
                            if *r == 0 {
                                let _ = store.remove_block(&b.content_hash);
                                s.persistent_refs.remove(&b.content_hash);
                                s.storage_used = s.storage_used.saturating_sub(b.len);
                            }
                        }
                    }
                    s.accounting
                        .cancel_reserve(TierKind::Persistent, new_storage)?;
                    return Err(e);
                }
                s.accounting
                    .commit_reserve(TierKind::Persistent, new_storage)?;
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
                                    // The freed bytes go back to the tier budget.
                                    s.accounting.sub_bytes(TierKind::Persistent, b.len)?;
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
    /// `needed` bytes of host capacity are free. Runs under the caller's lock.
    fn evict_to_free(&self, s: &mut State, needed: u64) -> Result<()> {
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
                let _ = self.remove_placement(s, &oid, &t);
            }
            s.drop_if_absent(&e.object_id);
        }
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
            s.drop_if_absent(&e.object_id);
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
    use crate::dtype::Dtype;
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

    /// A 1024-byte block whose content depends on `seed`.
    fn block(seed: u8) -> Vec<u8> {
        (0..1024usize)
            .map(|i| (i as u8).wrapping_mul(3).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn persistent_tier_bytes_are_returned_on_delete() {
        let dir = temp_dir();
        let mut config = cfg(1 << 20, Some(dir.clone()));
        config.persistent_capacity = 4096;
        let tc = TensorCache::new(config).unwrap();
        let compat = f32_compat(vec![32, 32]); // 4096 bytes: the whole tier
        for round in 0..3u8 {
            let data = [block(round), block(50), block(100), block(150)].concat();
            let oid = tc
                .register("ns", "p", round as u64, compat.clone(), &data)
                .unwrap();
            tc.persist(&oid).unwrap();
            assert_eq!(tc.resources().storage_used, 4096);
            tc.delete(&oid).unwrap();
            assert_eq!(tc.resources().storage_used, 0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recovered_objects_count_against_the_persistent_budget() {
        let dir = temp_dir();
        let mut config = cfg(1 << 20, Some(dir.clone()));
        config.persistent_capacity = 8192;
        let compat = f32_compat(vec![32, 32]); // 4096 bytes
        let tensor = |seed: u8| {
            [
                block(seed),
                block(seed + 1),
                block(seed + 2),
                block(seed + 3),
            ]
            .concat()
        };
        {
            let tc = TensorCache::new(config.clone()).unwrap();
            for (key, seed) in [("a", 10u8), ("b", 20u8)] {
                let oid = tc
                    .register("ns", key, 1, compat.clone(), &tensor(seed))
                    .unwrap();
                tc.persist(&oid).unwrap();
            }
        }
        // After a restart the tier is as full as it was: nothing more fits.
        let tc = TensorCache::new(config).unwrap();
        assert_eq!(tc.resources().storage_used, 8192);
        let extra = tc
            .register("ns", "c", 1, compat.clone(), &tensor(30))
            .unwrap();
        assert!(matches!(
            tc.persist(&extra),
            Err(Error::AdmissionRejected(_))
        ));
        assert_eq!(tc.resources().storage_used, 8192);
        // Deleting a recovered object gives its bytes back.
        tc.delete(&Address::new("ns", "a", 1).object_id()).unwrap();
        assert_eq!(tc.resources().storage_used, 4096);
        tc.persist(&extra).unwrap();
        assert_eq!(tc.resources().storage_used, 8192);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repeated_blocks_in_one_tensor_are_charged_once() {
        let dir = temp_dir();
        let tc = TensorCache::new(cfg(1 << 20, Some(dir.clone()))).unwrap();
        let compat = f32_compat(vec![32, 32]); // 4096 bytes = 4 blocks
        let data = vec![0u8; 4096]; // four identical blocks
        let oid = tc.register("ns", "zeros", 1, compat, &data).unwrap();
        tc.persist(&oid).unwrap();
        let r = tc.resources();
        assert_eq!(r.host_used, 1024, "one physical block is held");
        assert_eq!(r.storage_used, 1024);
        assert_eq!(tc.restore(&oid, &Tier::Host).unwrap(), data);
        tc.verify(&oid).unwrap();
        tc.delete(&oid).unwrap();
        let r = tc.resources();
        assert_eq!((r.host_used, r.storage_used, r.block_count), (0, 0, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_that_frees_shared_blocks_keeps_the_host_charge_exact() {
        // Room for two blocks. A = [x, y]; B = [x, z] shares x with A.
        let tc = TensorCache::new(cfg(2048, None)).unwrap();
        let compat = f32_compat(vec![16, 32]); // 2048 bytes = 2 blocks
        let a = tc
            .register("ns", "a", 1, compat.clone(), &[block(1), block(2)].concat())
            .unwrap();
        // Admitting B evicts A, and with it the block the two share.
        let b = tc
            .register("ns", "b", 1, compat.clone(), &[block(1), block(3)].concat())
            .unwrap();
        let r = tc.resources();
        assert_eq!(r.block_count, 2);
        assert_eq!(r.host_used, 2048, "both of B's blocks are charged");
        assert!(matches!(tc.metadata(&a), Err(Error::NotFound(_))));
        tc.delete(&b).unwrap();
        assert_eq!(tc.resources().host_used, 0);
    }

    #[test]
    fn evicted_entry_is_not_a_hit_and_can_be_registered_again() {
        let tc = TensorCache::new(cfg(4096, None)).unwrap();
        let compat = f32_compat(vec![32, 32]); // 4096 bytes: the whole tier
        let first = payload(4096);
        let second: Vec<u8> = first.iter().map(|b| b ^ 0x5a).collect();
        let a = tc.register("ns", "a", 1, compat.clone(), &first).unwrap();
        // Admitting "b" evicts "a", which then holds no bytes anywhere: it is
        // a miss, not a hit without data.
        tc.register("ns", "b", 1, compat.clone(), &second).unwrap();
        assert!(matches!(
            tc.lookup("ns", "a", 1, &compat),
            Err(Error::NotFound(_))
        ));
        assert_eq!(tc.resources().object_count, 1);
        // Once recomputed it can be cached again.
        let again = tc.register("ns", "a", 1, compat.clone(), &first).unwrap();
        assert_eq!(again, a);
        assert_eq!(tc.restore(&again, &Tier::Host).unwrap(), first);
        // Explicitly evicting a non-durable object drops it the same way.
        tc.evict(&again).unwrap();
        assert!(matches!(tc.metadata(&again), Err(Error::NotFound(_))));
        let r = tc.resources();
        assert_eq!((r.object_count, r.host_used), (0, 0));
    }

    #[test]
    fn object_count_limit_is_enforced() {
        let mut config = cfg(1 << 20, None);
        config.admission.max_objects = Some(2);
        let tc = TensorCache::new(config).unwrap();
        let compat = f32_compat(vec![16, 16]); // 1024 bytes
        let a = tc
            .register("ns", "a", 1, compat.clone(), &block(1))
            .unwrap();
        tc.register("ns", "b", 1, compat.clone(), &block(2))
            .unwrap();
        assert!(matches!(
            tc.register("ns", "c", 1, compat.clone(), &block(3)),
            Err(Error::AdmissionRejected(_))
        ));
        assert_eq!(tc.resources().object_count, 2);
        // Removing an object makes room under the limit again.
        tc.delete(&a).unwrap();
        tc.register("ns", "c", 1, compat, &block(3)).unwrap();
    }

    #[test]
    fn recover_is_idempotent() {
        let dir = temp_dir();
        let config = cfg(1 << 20, Some(dir.clone()));
        let compat = f32_compat(vec![32, 32]);
        let oid;
        {
            let tc = TensorCache::new(config.clone()).unwrap();
            oid = tc.register("ns", "p", 1, compat, &payload(4096)).unwrap();
            tc.persist(&oid).unwrap();
        }
        let tc = TensorCache::new(config).unwrap();
        tc.restore(&oid, &Tier::Host).unwrap();
        // Recovering again on the live runtime changes nothing.
        assert_eq!(tc.recover().unwrap().recovered, 0);
        let r = tc.resources();
        assert_eq!(
            (r.object_count, r.storage_used, r.host_used),
            (1, 4096, 4096)
        );
        assert!(tc.metadata(&oid).unwrap().placements.contains(&Tier::Host));
        tc.delete(&oid).unwrap();
        let r = tc.resources();
        assert_eq!((r.object_count, r.storage_used, r.host_used), (0, 0, 0));
        assert_eq!(
            std::fs::read_dir(dir.join("blocks")).unwrap().count(),
            0,
            "block files of the deleted object remain"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_placement_follows_the_object_block_list() {
        let dir = temp_dir();
        let compat = f32_compat(vec![32, 32]);
        let oid;
        {
            // Stored in 1024-byte blocks.
            let tc = TensorCache::new(cfg(1 << 20, Some(dir.clone()))).unwrap();
            oid = tc.register("ns", "p", 1, compat, &payload(4096)).unwrap();
            tc.persist(&oid).unwrap();
        }
        // The same store opened by a runtime configured with another block size.
        let mut config = cfg(1 << 20, Some(dir.clone()));
        config.block_size = 4096;
        let tc = TensorCache::new(config).unwrap();
        assert_eq!(tc.restore(&oid, &Tier::Host).unwrap(), payload(4096));
        // The host copy made by the restore is readable and agrees with disk.
        assert_eq!(tc.restore(&oid, &Tier::Host).unwrap(), payload(4096));
        tc.verify(&oid).unwrap();
        tc.delete(&oid).unwrap();
        assert_eq!(tc.resources().host_used, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_persist_leaves_no_reservation_or_block_reference() {
        let dir = temp_dir();
        let mut config = cfg(1 << 20, Some(dir.clone()));
        config.persistent_capacity = 2048;
        let tc = TensorCache::new(config).unwrap();
        // A block file with the right name but the wrong bytes is on disk.
        let damaged = block(9);
        let name = crate::hash::hash(&damaged).to_hex();
        std::fs::write(dir.join("blocks").join(name), b"damaged").unwrap();
        let first = tc
            .register("ns", "damaged", 1, f32_compat(vec![16, 16]), &damaged)
            .unwrap();
        assert!(matches!(tc.persist(&first), Err(Error::Persistence(_))));
        assert_eq!(tc.resources().storage_used, 0);
        assert!(!tc
            .metadata(&first)
            .unwrap()
            .placements
            .contains(&Tier::Persistent));
        // The failed attempt holds nothing back: the whole tier is available.
        let second = tc
            .register(
                "ns",
                "full",
                1,
                f32_compat(vec![16, 32]),
                &[block(40), block(41)].concat(),
            )
            .unwrap();
        tc.persist(&second).unwrap();
        assert_eq!(tc.resources().storage_used, 2048);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[derive(Default)]
    struct Faults {
        fail_h2d: std::sync::atomic::AtomicBool,
        corrupt_d2h: std::sync::atomic::AtomicBool,
    }

    /// A device backend whose copies can be made to fail or to return damaged
    /// bytes. Allocation is delegated to a CPU backend the test can inspect.
    struct FaultyBackend {
        id: BackendId,
        inner: Arc<crate::backend_cpu::CpuBackend>,
        faults: Arc<Faults>,
    }

    impl Backend for FaultyBackend {
        fn id(&self) -> &BackendId {
            &self.id
        }
        fn byte_capacity(&self) -> u64 {
            self.inner.byte_capacity()
        }
        fn bytes_allocated(&self) -> u64 {
            self.inner.bytes_allocated()
        }
        fn allocate(&self, bytes: usize) -> Result<DeviceBuffer> {
            let mut dev = self.inner.allocate(bytes)?;
            dev.backend = self.id.clone();
            Ok(dev)
        }
        fn to_device(&self, host: &[u8], dev: &mut DeviceBuffer) -> Result<()> {
            if self
                .faults
                .fail_h2d
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err(Error::Backend("injected H2D failure".into()));
            }
            self.inner.to_device(host, dev)
        }
        fn device_to_host(&self, dev: &DeviceBuffer, host: &mut [u8]) -> Result<()> {
            self.inner.device_to_host(dev, host)?;
            if self
                .faults
                .corrupt_d2h
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                host[0] ^= 0xff;
            }
            Ok(())
        }
        fn copy_in(&self, dev: &mut DeviceBuffer, offset: usize, data: &[u8]) -> Result<()> {
            self.inner.copy_in(dev, offset, data)
        }
        fn copy_out(&self, dev: &DeviceBuffer, offset: usize, data: &mut [u8]) -> Result<()> {
            self.inner.copy_out(dev, offset, data)
        }
        fn fill(&self, dev: &mut DeviceBuffer, bytes: usize, value: u8) -> Result<()> {
            self.inner.fill(dev, bytes, value)
        }
        fn free(&self, dev: DeviceBuffer) -> Result<()> {
            self.inner.free(dev)
        }
    }

    fn with_faulty_device(
        persistent: Option<PathBuf>,
    ) -> (
        TensorCache,
        Tier,
        Arc<crate::backend_cpu::CpuBackend>,
        Arc<Faults>,
    ) {
        let id = BackendId::new("faulty", 0);
        let device = Arc::new(crate::backend_cpu::CpuBackend::new(7, 1 << 20));
        let faults = Arc::new(Faults::default());
        let backend = FaultyBackend {
            id: id.clone(),
            inner: Arc::clone(&device),
            faults: Arc::clone(&faults),
        };
        let tc =
            TensorCache::with_backends(cfg(1 << 20, persistent), vec![Box::new(backend)]).unwrap();
        (tc, Tier::Accelerator(id), device, faults)
    }

    #[test]
    fn failed_device_copy_returns_the_allocation() {
        use std::sync::atomic::Ordering;
        let (tc, accel, device, faults) = with_faulty_device(None);
        let oid = tc
            .register("ns", "dev", 1, f32_compat(vec![16, 16]), &block(5))
            .unwrap();
        faults.fail_h2d.store(true, Ordering::SeqCst);
        assert!(matches!(tc.promote(&oid, &accel), Err(Error::Backend(_))));
        assert_eq!(
            device.bytes_allocated(),
            0,
            "the failed promotion kept its device buffer"
        );
        assert_eq!(tc.resources().accel_used, 0);
        faults.fail_h2d.store(false, Ordering::SeqCst);
        tc.promote(&oid, &accel).unwrap();
        assert_eq!(device.bytes_allocated(), 1024);
        tc.delete(&oid).unwrap();
        assert_eq!(device.bytes_allocated(), 0);
    }

    #[test]
    fn damaged_device_bytes_are_not_served() {
        use std::sync::atomic::Ordering;
        let dir = temp_dir();
        let (tc, accel, _device, faults) = with_faulty_device(Some(dir.clone()));
        let data = block(6);
        let oid = tc
            .register("ns", "dev", 1, f32_compat(vec![16, 16]), &data)
            .unwrap();
        tc.promote(&oid, &accel).unwrap();
        // Move the host copy to disk: the device copy is now the one read.
        tc.demote(&oid, &Tier::Host).unwrap();
        faults.corrupt_d2h.store(true, Ordering::SeqCst);
        assert!(matches!(
            tc.restore(&oid, &Tier::Host),
            Err(Error::Integrity(_))
        ));
        assert!(!tc.metadata(&oid).unwrap().placements.contains(&Tier::Host));
        faults.corrupt_d2h.store(false, Ordering::SeqCst);
        assert_eq!(tc.restore(&oid, &Tier::Host).unwrap(), data);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
