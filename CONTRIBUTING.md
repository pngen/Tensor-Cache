# Contributing

Thank you for contributing to `Tensor Cache`. Please read this before opening a
pull request.

## Design invariants (must hold)

- **No unsafe code in the core.** The `tensorcache` crate is `#![forbid(unsafe_code)]`.
  Any unsafe FFI must live in an isolated crate/module behind a narrow safe
  abstraction (the CUDA crate is the model).
- **No false reuse.** Compatibility identity is canonical and hashed; a reuse
  requires an exact `compat_id` match. Do not add heuristic or fuzzy
  compatibility.
- **No unbounded growth.** Admission and eviction must remain bounded and
  deterministic.
- **No deadlocks.** The local runtime uses a single mutex and never re-enters it.
  Internal helpers snapshot, release the guard, then act. Do not re-acquire the
  lock, call persistence/snapshot helpers while holding a guard they touch, join a
  worker while holding a required lock, or wait on a channel while holding a
  required lock.
- **Exact accounting.** Bytes must never go negative or overshoot; placeholders are
  never invented.
- **Deterministic where semantics permit.** Eviction order, planner decisions,
  and admission decisions must be reproducible.

## Dependency policy

The core should remain zero-dependency. Add a dependency only with strong
justification; the CUDA crate is the exception (and it loads the runtime
dynamically). No telemetry, analytics, network, cloud, account or external
service dependency.

## Validation before submit

Run all validations plainly (never add timeouts):

```
cargo fmt --check
cargo build
cargo build --release
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p tensorcache-cli --features cuda --all-targets -- -D warnings
cargo bench --bench runtime --package tensorcache
cargo run --example cache_hit --package tensorcache
```

Ensure all examples run and no warnings are emitted.

## Style

- Follow rustfmt.
- Write meaningful tests around real correctness risks, not inflated counts.
- Keep documentation tied to what was actually validated.
- Do not market unsupported functionality; if a backend or transport is not
  implemented, do not claim it is.
