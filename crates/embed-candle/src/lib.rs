//! Candle implementation of [`InferenceEngine`].
//!
//! Status: skeleton. The structure (engine + trait impl) is in place; the real
//! all-MiniLM-L6-v2 forward pass lands in Phase 0. Keep every `candle_*` type
//! confined to this crate so the framework boundary stays clean.

use anyhow::{bail, Result};
use embed_core::{Embedding, InferenceEngine};

/// Candle-backed embedding engine (CPU backend, per the desktop-only scope).
pub struct CandleEngine {
    name: String,
    // TODO(phase-0): model, tokenizer, and device live here.
}

impl CandleEngine {
    /// Load weights + tokenizer from `model_dir`.
    pub fn load(_model_dir: &str) -> Result<Self> {
        // TODO(phase-0): load safetensors + tokenizer.json, build BERT, CPU device.
        Ok(Self {
            name: "candle-cpu".to_string(),
        })
    }
}

impl InferenceEngine for CandleEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn embed(&self, _text: &str) -> Result<Embedding> {
        bail!("CandleEngine::embed not yet implemented (Phase 0)")
    }
}
