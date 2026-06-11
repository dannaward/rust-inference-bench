//! Build-time codegen: burn-import turns the all-MiniLM ONNX export into a Burn
//! model. The ONNX file is large and gitignored, so it must be fetched first
//! (see scripts/fetch-model.sh); we fail loudly if it is missing.

use burn_import::onnx::ModelGen;

fn main() {
    let onnx = "artifacts/model.onnx";
    if !std::path::Path::new(onnx).exists() {
        panic!(
            "missing {onnx} — run `scripts/fetch-model.sh` to download the ONNX weights \
             before building embed-burn"
        );
    }
    ModelGen::new().input(onnx).out_dir("model/").run_from_script();
    println!("cargo:rerun-if-changed={onnx}");
}
