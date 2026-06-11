//! Burn implementation of [`InferenceEngine`].
//!
//! Status: skeleton. The structure (engine + trait impl) is in place; the real
//! all-MiniLM-L6-v2 forward pass lands in Phase 0. Keep every `burn` type
//! confined to this crate so the framework boundary stays clean.
//!
//! Note: do NOT use the `burn-candle` backend here -- that would turn the
//! comparison into Candle-vs-Candle. Use a Burn first-party backend (ndarray
//! for the desktop CPU baseline).

use anyhow::{bail, Result};
use embed_core::{Embedding, InferenceEngine};

/// Burn-backed embedding engine (ndarray backend for the desktop CPU baseline).
pub struct BurnEngine {
    name: String,
    // TODO(phase-0): model, tokenizer, and device live here.
}

impl BurnEngine {
    /// Load weights + tokenizer from `model_dir`.
    pub fn load(_model_dir: &str) -> Result<Self> {
        // TODO(phase-0): import model (ONNX -> burn-import or hand-built BERT),
        // load tokenizer.json, set ndarray backend device.
        Ok(Self {
            name: "burn-ndarray".to_string(),
        })
    }
}

impl InferenceEngine for BurnEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn embed(&self, _text: &str) -> Result<Embedding> {
        bail!("BurnEngine::embed not yet implemented (Phase 0)")
    }
}
