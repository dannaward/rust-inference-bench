# candle-vs-burn-bench

A reproducible benchmark comparing the **Candle** and **Burn** Rust ML frameworks
on a sentence-embedding inference workload, to inform the NAHPU framework choice.

## Scope (intentional constraints)

- **Desktop only.** No mobile / on-device targets. The Flutter (FFI) integration
  layer adds a *framework-agnostic constant*, so it does not change the relative
  ranking of the two frameworks — a standalone Rust benchmark is sufficient for
  the framework decision. (Mobile would change the picture via per-platform
  backend availability, but that is out of scope here.)
- **Inference only.** No training / fine-tuning.
- **One model:** `all-MiniLM-L6-v2` (22M params, 384-dim sentence embeddings).
- **CPU baseline first** (Candle CPU vs Burn `ndarray`); GPU is a later phase.

## Why a benchmark at all

No public source does a rigorous, same-model/same-hardware head-to-head of Candle
vs Burn — they only report each framework against PyTorch separately. This repo
produces that comparison for NAHPU's actual workload.

## Architecture

Every framework lives behind the `InferenceEngine` trait in `embed-core`. Nothing
outside `embed-candle` / `embed-burn` touches a framework type — only `&str` in,
`Vec<f32>` out cross the boundary. This keeps the comparison fair and makes a
future Candle->Burn migration one new impl rather than a rewrite.

```
crates/
  embed-core/     trait + types + cosine util + corpus loader (no framework deps)
  embed-candle/   CandleEngine   (skeleton; Phase 0 fills in the forward pass)
  embed-burn/     BurnEngine     (skeleton; ndarray backend, NOT burn-candle)
runner/
  src/main.rs        Phase 0 parity check (cosine similarity between engines)
  benches/embedding.rs  Criterion latency + throughput
data/
  corpus.sample.txt  synthetic specimen-like text (no real NAHPU data)
  models/            (gitignored) model weights fetched locally
```

## Phases

0. **Parity** — confirm Candle and Burn produce equivalent embeddings
   (min cosine similarity > 0.999). Prerequisite for trusting any speed number.
1. **CPU ↔ CPU** — Candle (CPU) vs Burn (`ndarray`). The desktop baseline.
2. **GPU ↔ GPU** — Candle (Metal) vs Burn (`wgpu`). Only if acceleration matters.
3. **Secondary metrics** — peak memory (RSS), binary size, cold start, build time.

## Metrics & fairness rules

- Report **distributions** (median + p95/p99), never a single number.
- **Warm up** before measuring; measure cold start separately.
- Hold constant: model weights, tokenizer (`tokenizers` crate, excluded from
  timing), input corpus, precision (f32), thread count, release build flags.
- Pin threads when running benches: `RAYON_NUM_THREADS=1` (+ BLAS) for the
  controlled baseline, then scale deliberately.
- **Do not** use Burn's `burn-candle` backend — it would make this Candle vs Candle.

## Running

```sh
# Phase 0 parity (skeletons currently error out until the forward passes land)
cargo run -p runner -- data/corpus.sample.txt data/models/all-MiniLM-L6-v2

# Benchmarks (once engines are implemented)
RAYON_NUM_THREADS=1 cargo bench -p runner

# Unit tests
cargo test
```

## Status

- **Phase 0 (parity): done** — Candle (safetensors) and Burn (ONNX import) agree
  to min cosine **1.000000**.
- **Phase 1 (CPU latency + throughput): done** — see [REPORT.md](REPORT.md).
  Recommendation: **Candle** for the desktop embedding workload.
- **Secondary metrics: done** — Candle also wins cold start, RSS, binary size.
- **Phase 2 (GPU): done** — Candle Metal beats Burn wgpu by 4–7.8× (`REPORT.md`).
