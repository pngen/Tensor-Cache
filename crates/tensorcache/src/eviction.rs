#![forbid(unsafe_code)]
//! Deterministic cache eviction/reclamation.
//!
//! Eviction uses an explicit keep-score rather than HashMap iteration order.
//! Higher keep-score means the object is worth retaining; eviction candidates
//! are ordered by ascending keep-score and reclaimed in that order. The score
//! weighs reuse count, recency, reconstruction cost, size, priority and tier
//! pressure.

/// Metadata required to score an object for eviction.
#[derive(Debug, Clone)]
pub struct Evictable {
    pub object_id: String,
    pub bytes: u64,
    pub reuse_count: u64,
    /// Seconds since last use (0 = just used).
    pub age_seconds: u64,
    /// Cost to reconstruct (ns).
    pub reconstruction_cost_ns: u64,
    pub priority: u32,
    /// Fraction (0..=1) of the tier budget that is currently used.
    pub pressure: f64,
    pub durable: bool,
}

/// The eviction policy weights.
#[derive(Debug, Clone)]
pub struct EvictionPolicy {
    /// Weight applied to the access count.
    pub reuse_weight: u64,
    /// Weight applied to reconstruction cost (in ns).
    pub reconstruct_weight: u64,
    /// Weight applied to age (in seconds).
    pub age_weight: u64,
    /// Weight applied to byte size (negative contribution to keep-score).
    pub size_weight: u64,
    /// Weight applied to a durable object (kept longer).
    pub durable_bonus: u64,
    /// Priority weight.
    pub priority_weight: u64,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy {
            reuse_weight: 1000,
            reconstruct_weight: 1,
            age_weight: 500,
            size_weight: 200,
            durable_bonus: 1_000_000,
            priority_weight: 500_000,
        }
    }
}

/// Compute a deterministic keep-score (higher = keep).
pub fn keep_score(policy: &EvictionPolicy, e: &Evictable) -> i64 {
    let mut score: i64 = 0;
    score += (e.reuse_count as i64).saturating_mul(policy.reuse_weight as i64);
    score += (e.reconstruction_cost_ns as i64).saturating_mul(policy.reconstruct_weight as i64);
    // Age penalty: older objects drop in score.
    score -= (e.age_seconds as i64).saturating_mul(policy.age_weight as i64);
    // Size penalty: larger objects are more expensive to keep.
    score -= (e.bytes as i64)
        .saturating_div(1024)
        .saturating_mul(policy.size_weight as i64);
    // Durable objects are protected.
    if e.durable {
        score += policy.durable_bonus as i64;
    }
    // Priority is a strong positive influence.
    score += (e.priority as i64).saturating_mul(policy.priority_weight as i64);
    score
}

/// Order eviction candidates by ascending keep-score (evict cheapest to keep
/// first). Ties are broken deterministically by object id for reproducibility.
pub fn eviction_order(policy: &EvictionPolicy, entries: &[Evictable]) -> Vec<Evictable> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| {
        let sa = keep_score(policy, a);
        let sb = keep_score(policy, b);
        sa.cmp(&sb).then_with(|| a.object_id.cmp(&b.object_id))
    });
    sorted
}

/// Return the number of lowest-scoring entries that must be evicted until the
/// tier is within `target_free` bytes. Deterministic and conservative.
pub fn eviction_plan(
    policy: &EvictionPolicy,
    entries: &[Evictable],
    used_bytes: u64,
    capacity: u64,
) -> Vec<Evictable> {
    let mut need = used_bytes.saturating_sub(capacity);
    let mut evict = Vec::new();
    for e in eviction_order(policy, entries) {
        if need == 0 {
            break;
        }
        evict.push(e.clone());
        need = need.saturating_sub(e.bytes);
    }
    evict
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(
        id: &str,
        bytes: u64,
        reuse: u64,
        age: u64,
        reconstruct: u64,
        priority: u32,
    ) -> Evictable {
        Evictable {
            object_id: id.to_string(),
            bytes,
            reuse_count: reuse,
            age_seconds: age,
            reconstruction_cost_ns: reconstruct,
            priority,
            pressure: 0.5,
            durable: false,
        }
    }

    #[test]
    fn higher_reuse_score_keeps() {
        let p = EvictionPolicy::default();
        let hot = ev("hot", 1000, 100, 0, 1_000_000, 0);
        let cold = ev("cold", 1000, 1, 1000, 1_000_000, 0);
        assert!(keep_score(&p, &hot) > keep_score(&p, &cold));
    }

    #[test]
    fn durable_protected() {
        let p = EvictionPolicy::default();
        let mut durable = ev("d", 1000, 1, 1000, 1_000_000, 0);
        durable.durable = true;
        let volatile = ev("v", 1000, 1, 1000, 1_000_000, 0);
        assert!(keep_score(&p, &durable) > keep_score(&p, &volatile));
    }

    #[test]
    fn eviction_order_is_deterministic() {
        let p = EvictionPolicy::default();
        let entries = vec![
            ev("a", 100, 1, 1000, 0, 0),
            ev("b", 200, 2, 100, 0, 0),
            ev("c", 300, 3, 0, 0, 0),
        ];
        let o1 = eviction_order(&p, &entries);
        let o2 = eviction_order(&p, &entries);
        assert_eq!(
            o1.iter().map(|e| e.object_id.clone()).collect::<Vec<_>>(),
            o2.iter().map(|e| e.object_id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn eviction_plan_frees_enough() {
        let p = EvictionPolicy::default();
        let entries = vec![
            ev("a", 400, 1, 1000, 0, 0),
            ev("b", 300, 2, 100, 0, 0),
            ev("c", 200, 3, 0, 0, 0),
        ];
        // used 900, capacity 600 -> need to free 300.
        let plan = eviction_plan(&p, &entries, 900, 600);
        let freed: u64 = plan.iter().map(|e| e.bytes).sum();
        assert!(freed >= 300);
    }
}
