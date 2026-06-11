//! Shared abstractions for the Candle-vs-Burn embedding benchmark.
//!
//! The whole point of this crate is *framework isolation*: nothing here depends
//! on Candle or Burn. Each framework lives behind [`InferenceEngine`], so the
//! benchmark harness, the parity check, and (later) any integration code stay
//! identical regardless of which framework produced the vectors. A future
//! Candle -> Burn migration is then one new impl rather than a rewrite.

use std::path::Path;

use anyhow::{Context, Result};

/// A single sentence embedding. all-MiniLM-L6-v2 produces 384 dimensions.
pub type Embedding = Vec<f32>;

/// Expected embedding dimensionality for the benchmark model (all-MiniLM-L6-v2).
pub const EMBED_DIM: usize = 384;

/// A swappable text-embedding backend.
///
/// Keep every framework type (tensors, devices, modules) *inside* the
/// implementor -- only domain types (`&str` in, [`Embedding`] out) cross this
/// boundary.
pub trait InferenceEngine {
    /// Short identifier used in reports, e.g. `"candle-cpu"`.
    fn name(&self) -> &str;

    /// Embed a single piece of text.
    fn embed(&self, text: &str) -> Result<Embedding>;

    /// Embed a batch. The default loops; override when the framework can batch
    /// more efficiently (that gap is exactly what the throughput test measures).
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Cosine similarity between two vectors. Used by the Phase 0 parity check to
/// confirm Candle and Burn produce equivalent embeddings before any speed
/// number is trusted.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Load a benchmark corpus: one text per non-empty line.
///
/// Keep the file static and committed (synthetic / public text only -- no real
/// NAHPU specimen data) so results stay reproducible.
pub fn load_corpus(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading corpus from {}", path.display()))?;
    Ok(raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
