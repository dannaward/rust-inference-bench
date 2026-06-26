//! Cross-platform EP diagnostic (worklog 08): on the box you're testing, prints
//! (1) which ORT execution providers the loaded runtime actually exposes, and
//! (2) a parity check — every compiled-in EP must match ort-cpu within the same
//! cosine>0.999 tolerance the main parity gate uses, so EP speed numbers are
//! trustworthy. Run e.g.:
//!   cargo run -p embed-ort --example ep_probe --features "xnnpack coreml"
use embed_core::{cosine_similarity, InferenceEngine};
use ort::execution_providers::ExecutionProvider;

fn main() -> anyhow::Result<()> {
    // (1) availability: GetAvailableProviders on the loaded ORT binary.
    let cpu = ort::execution_providers::CPUExecutionProvider::default();
    println!("CPU     available: {:?}", cpu.is_available());
    let xnn = ort::execution_providers::XNNPACKExecutionProvider::default();
    println!("XNNPACK available: {:?}", xnn.is_available());
    #[cfg(feature = "coreml")]
    {
        let cm = ort::execution_providers::CoreMLExecutionProvider::default();
        println!("CoreML  available: {:?}", cm.is_available());
    }

    // (2) parity: reference = ort-cpu, compared against each compiled-in EP.
    let dir = "data/models/all-MiniLM-L6-v2";
    let texts: Vec<String> = [
        "Dark brown iris.",
        "Adult male with reddish-brown dorsal fur.",
        "Collected near a montane stream at 1850 meters elevation.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let reference = embed_ort::OrtEngine::load(dir)?;
    let ref_out = reference.embed_batch(&texts)?;

    let mut peers: Vec<embed_ort::OrtEngine> = Vec::new();
    #[cfg(feature = "xnnpack")]
    peers.push(embed_ort::OrtEngine::load_xnnpack(dir)?);
    #[cfg(feature = "coreml")]
    peers.push(embed_ort::OrtEngine::load_coreml(dir)?);

    let mut overall_min = f32::MAX;
    for peer in &peers {
        let out = peer.embed_batch(&texts)?;
        let min_sim = ref_out
            .iter()
            .zip(&out)
            .map(|(a, b)| cosine_similarity(a, b))
            .fold(f32::MAX, f32::min);
        overall_min = overall_min.min(min_sim);
        println!("{} vs ort-cpu: min cosine = {min_sim:.6}", peer.name());
    }
    if peers.is_empty() {
        println!("(no EP features enabled — nothing to parity-check)");
    } else {
        println!(
            "EP parity {} (worst min cosine = {overall_min:.6})",
            if overall_min > 0.999 { "PASS" } else { "FAIL" }
        );
    }
    Ok(())
}
