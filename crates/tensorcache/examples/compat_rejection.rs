//! Example 2: compatibility rejection.
//!
//! The same logical key requested with an incompatible dtype/layout/model must
//! be rejected, not silently reused. A false reuse hit is a correctness
//! failure.

mod common;
use tensorcache::error::Error;
use tensorcache::runtime::TensorCache;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tc = TensorCache::new(common::config(1 << 24, None))?;
    let f32_key = common::f32(vec![32, 32], "model-a");

    tc.register(
        "ns",
        "attn.q_proj",
        1,
        f32_key.clone(),
        &common::payload(4096),
    )?;
    println!("registered attn.q_proj with model-a / f32 / [32,32]");

    // Same key, but a different dtype -> must be rejected.
    let mut f16_key = f32_key.clone();
    f16_key.dtype = tensorcache::dtype::Dtype::F16;
    let r = tc.lookup("ns", "attn.q_proj", 1, &f16_key);
    assert!(matches!(r, Err(Error::Compatibility(_))));
    println!("rejected: dtype mismatch (f32 cached, f16 requested)");

    // Same key, different model -> must be rejected.
    let mut other_model = f32_key.clone();
    other_model.model = Some("model-b".into());
    let r = tc.lookup("ns", "attn.q_proj", 1, &other_model);
    assert!(matches!(r, Err(Error::Compatibility(_))));
    println!("rejected: model identity mismatch");

    // Same key + same compat -> accepted.
    assert!(tc.lookup("ns", "attn.q_proj", 1, &f32_key).is_ok());
    println!("accepted: exact compatibility");
    Ok(())
}
