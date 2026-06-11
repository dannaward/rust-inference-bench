//! Phase 0 parity check: confirm Candle and Burn produce equivalent embeddings
//! for the same input. Until this passes, any speed comparison is apples-to-
//! oranges, so this runs first.
//!
//! Usage: `cargo run -p runner -- [corpus_path] [model_dir]`

use anyhow::Result;
use embed_burn::BurnEngine;
use embed_candle::CandleEngine;
use embed_core::{cosine_similarity, load_corpus, Embedding, InferenceEngine};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let corpus_path = args
        .next()
        .unwrap_or_else(|| "data/corpus.sample.txt".to_string());
    let model_dir = args
        .next()
        .unwrap_or_else(|| "data/models/all-MiniLM-L6-v2".to_string());

    let corpus = load_corpus(&corpus_path)?;
    println!("Loaded {} sentences from {corpus_path}", corpus.len());

    let engines: Vec<Box<dyn InferenceEngine>> = vec![
        Box::new(CandleEngine::load(&model_dir)?),
        Box::new(BurnEngine::load(&model_dir)?),
    ];

    // Embed the whole corpus with each engine that is actually implemented.
    let mut outputs: Vec<(String, Vec<Embedding>)> = Vec::new();
    for engine in &engines {
        match engine.embed_batch(&corpus) {
            Ok(v) => outputs.push((engine.name().to_string(), v)),
            Err(e) => println!("  [skip] {}: {e}", engine.name()),
        }
    }

    if outputs.len() < 2 {
        println!(
            "\nNeed >=2 working engines for a parity check. \
             Implement the forward passes (Phase 0), then re-run."
        );
        return Ok(());
    }

    // Compare the first two engines sentence-by-sentence.
    let (name_a, a) = &outputs[0];
    let (name_b, b) = &outputs[1];
    let min_sim = a
        .iter()
        .zip(b)
        .map(|(x, y)| cosine_similarity(x, y))
        .fold(f32::MAX, f32::min);

    println!("\n{name_a} vs {name_b}: min cosine similarity = {min_sim:.6}");
    println!(
        "Parity {}",
        if min_sim > 0.999 { "PASS" } else { "FAIL" }
    );
    Ok(())
}
