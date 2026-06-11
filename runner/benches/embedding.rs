//! Criterion latency/throughput benchmark for the embedding engines.
//!
//! Methodology (see README):
//! - Criterion handles warm-up; cold start (model load + first inference) is a
//!   separate metric, not measured here.
//! - Report the distribution (median + p95), never a single number.
//! - Tokenization is shared across frameworks; keep it out of the framework
//!   comparison where possible.
//! - Pin thread counts before running: `RAYON_NUM_THREADS=1` (and BLAS threads)
//!   for a controlled single-thread baseline, then scale deliberately.
//!
//! Note: skipped until the engines are implemented -- `embed` currently errors,
//! so `engines()` returns an empty set and the groups are no-ops.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use embed_burn::BurnEngine;
use embed_candle::CandleEngine;
use embed_core::{load_corpus, InferenceEngine};

const MODEL_DIR: &str = "data/models/all-MiniLM-L6-v2";
const CORPUS: &str = "data/corpus.sample.txt";

/// Only return engines whose `embed` actually works, so the benchmark stays a
/// no-op while the impls are still skeletons.
fn engines() -> Vec<Box<dyn InferenceEngine>> {
    let mut v: Vec<Box<dyn InferenceEngine>> = Vec::new();
    if let Ok(e) = CandleEngine::load(MODEL_DIR) {
        if e.embed("warmup probe").is_ok() {
            v.push(Box::new(e));
        }
    }
    if let Ok(e) = BurnEngine::load(MODEL_DIR) {
        if e.embed("warmup probe").is_ok() {
            v.push(Box::new(e));
        }
    }
    v
}

fn bench_single(c: &mut Criterion) {
    let corpus = load_corpus(CORPUS).unwrap_or_default();
    let sample = corpus
        .first()
        .cloned()
        .unwrap_or_else(|| "a specimen note".to_string());

    let mut group = c.benchmark_group("embed_single");
    for engine in engines() {
        group.bench_with_input(
            BenchmarkId::from_parameter(engine.name()),
            &sample,
            |b, s| b.iter(|| engine.embed(s).expect("embed")),
        );
    }
    group.finish();
}

fn bench_batch(c: &mut Criterion) {
    let corpus = load_corpus(CORPUS).unwrap_or_default();
    if corpus.is_empty() {
        return;
    }
    let mut group = c.benchmark_group("embed_batch");
    group.throughput(Throughput::Elements(corpus.len() as u64));
    for engine in engines() {
        group.bench_with_input(
            BenchmarkId::from_parameter(engine.name()),
            &corpus,
            |b, texts| b.iter(|| engine.embed_batch(texts).expect("embed_batch")),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_single, bench_batch);
criterion_main!(benches);
