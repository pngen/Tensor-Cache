#![forbid(unsafe_code)]
//! Deterministic cost model for reuse economics.
//!
//! Costs are expressed as an integer number of nanoseconds so that the planner
//! is fully deterministic for a given configuration and input. Bandwidth,
//! latency and reconstruction parameters are explicit policy inputs; they are
//! NOT claimed to be portable microbenchmarks. On real hardware the operator
//! should tune the model to the observed tier characteristics.

use crate::tiers::{Tier, TierKind};

/// A configurable, deterministic cost model.
#[derive(Debug, Clone)]
pub struct CostModel {
    pub accel_read_bw: u64,
    pub accel_write_bw: u64,
    pub host_read_bw: u64,
    pub host_write_bw: u64,
    pub storage_read_bw: u64,
    pub storage_write_bw: u64,
    /// One-way transfer setup latency across a tier hop, in nanoseconds.
    pub transfer_latency_ns: u64,
    /// Restore-from-persistent setup latency, in nanoseconds.
    pub restore_latency_ns: u64,
    /// Base recompute cost, in nanoseconds.
    pub recompute_base_ns: u64,
    /// Additional recompute cost per byte, in nanoseconds.
    pub recompute_ns_per_byte: u64,
    /// Per-byte durable-write cost, in nanoseconds. Used to compare persisting
    /// an object against reconstructing it later.
    pub persist_ns_per_byte: u64,
}

impl Default for CostModel {
    fn default() -> Self {
        // Sensible, conservative defaults. Tune for real hardware.
        CostModel {
            accel_read_bw: 1_800_000_000_000, // 1.8 TB/s
            accel_write_bw: 1_800_000_000_000,
            host_read_bw: 20_000_000_000, // 20 GB/s
            host_write_bw: 20_000_000_000,
            storage_read_bw: 1_500_000_000, // 1.5 GB/s SSD
            storage_write_bw: 1_200_000_000,
            transfer_latency_ns: 20_000,  // 20 us
            restore_latency_ns: 250_000,  // 250 us
            recompute_base_ns: 5_000_000, // 5 ms
            recompute_ns_per_byte: 30,    // ~33 MB/s recompute throughput
            persist_ns_per_byte: 60,
        }
    }
}

impl CostModel {
    fn read_bw(&self, k: TierKind) -> u64 {
        match k {
            TierKind::Accelerator => self.accel_read_bw,
            TierKind::Host => self.host_read_bw,
            TierKind::Persistent => self.storage_read_bw,
        }
    }

    fn write_bw(&self, k: TierKind) -> u64 {
        match k {
            TierKind::Accelerator => self.accel_write_bw,
            TierKind::Host => self.host_write_bw,
            TierKind::Persistent => self.storage_write_bw,
        }
    }

    /// Nanoseconds required to move `bytes` at the given bandwidth.
    fn transfer_ns(bytes: u64, bandwidth: u64) -> u64 {
        if bandwidth == 0 {
            return u64::MAX;
        }
        let b = bytes as u128;
        let ns = b
            .saturating_mul(1_000_000_000)
            .checked_div(bandwidth as u128)
            .unwrap_or(u128::MAX);
        ns.min(u64::MAX as u128) as u64
    }

    /// Cost to transfer `bytes` from `src` to `dst`.
    pub fn transfer_cost_ns(&self, src: &Tier, dst: &Tier, bytes: u64) -> u64 {
        let sk = src.kind();
        let dk = dst.kind();
        let effective = self.read_bw(sk).min(self.write_bw(dk));
        self.transfer_latency_ns
            .saturating_add(Self::transfer_ns(bytes, effective))
    }

    /// Cost to restore `bytes` from persistent storage to `dst`.
    pub fn restore_cost_ns(&self, dst: &Tier, bytes: u64) -> u64 {
        let dk = dst.kind();
        let effective = self.storage_read_bw.min(self.write_bw(dk));
        self.restore_latency_ns
            .saturating_add(Self::transfer_ns(bytes, effective))
    }

    /// Cost to reconstruct/recompute `bytes`.
    pub fn reconstruct_cost_ns(&self, bytes: u64) -> u64 {
        self.recompute_base_ns.saturating_add(
            (bytes as u128)
                .saturating_mul(self.recompute_ns_per_byte as u128)
                .min(u64::MAX as u128) as u64,
        )
    }

    /// Cost to persist `bytes` durably.
    pub fn persist_cost_ns(&self, bytes: u64) -> u64 {
        (bytes as u128)
            .saturating_mul(self.persist_ns_per_byte as u128)
            .min(u64::MAX as u128) as u64
    }

    /// The cost of an in-place reuse (no movement, no reconstruction).
    pub const fn reuse_in_place_cost_ns(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_cheaper_than_recompute_for_small_bytes() {
        let m = CostModel::default();
        let bytes = 1024;
        let transfer = m.transfer_cost_ns(&Tier::Host, &Tier::Host, bytes);
        let recompute = m.reconstruct_cost_ns(bytes);
        assert!(transfer < recompute);
    }

    #[test]
    fn recompute_cheaper_than_transfer_when_transfer_is_slow() {
        // Tailor the model: very slow persistent write, cheap recompute.
        let m = CostModel {
            storage_write_bw: 1_000_000,
            recompute_ns_per_byte: 1,
            recompute_base_ns: 1_000,
            ..Default::default()
        };
        let bytes = 1_000_000;
        let transfer = m.transfer_cost_ns(&Tier::Host, &Tier::Persistent, bytes);
        let recompute = m.reconstruct_cost_ns(bytes);
        assert!(recompute < transfer);
    }

    #[test]
    fn zero_bandwidth_is_expensive_not_infinite_overflow() {
        let m = CostModel {
            host_read_bw: 0,
            ..Default::default()
        };
        let c = m.transfer_cost_ns(&Tier::Host, &Tier::Persistent, 100);
        assert!(c == u64::MAX || c > 0);
    }
}
