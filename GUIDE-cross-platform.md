# Cross-platform ONNX Runtime EP guide

How to build and run the `embed-ort` engine on each ONNX Runtime **execution
provider (EP)**, so we can measure inference acceleration on the hardware Candle
has no backend for — Intel iGPU, AMD GPU, and Android. Companion to worklog
`08-ort-cross-platform-ep-plan.md`.

The bench loads one model (`all-MiniLM-L6-v2`, same ONNX export as every other
engine) and reports latency + throughput. The `ep` mode runs an **ort-cpu
baseline plus whichever EPs you compiled in**, so every EP number is directly
comparable to CPU on the same box. Parity (cosine > 0.999) is gated by a separate
binary — see "Verify the EP first" below; the `ep` mode itself only measures.

## Verify the EP first (availability + parity)

Before trusting any speed number, confirm the EP is actually present in your ORT
build and produces the same vectors as ort-cpu. The `ep_probe` example does both
(`GetAvailableProviders` + a cosine parity check), with no Burn/Candle deps:

```sh
cargo run -p embed-ort --example ep_probe --features "xnnpack coreml"
# -> "XNNPACK available: Ok(true)" + "EP parity PASS (worst min cosine = 1.000000)"
```

The full parity binary also picks up the EP engine under the same `ep-*` feature:

```sh
cargo run --release -p runner --bin runner --features ep-xnnpack
```

## TL;DR — run the EP matrix

```sh
# Build with one or more ep-* features, then run `ep` mode:
RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench \
    --features ep-xnnpack -- ep
```

`ep-*` features (each maps to one ORT EP):

| Feature | EP | Target hardware | Builds on this Mac? |
|---|---|---|---|
| `ep-xnnpack` | XNNPACK | CPU/SIMD (ARM NEON, x86 AVX) — portable | ✅ yes |
| `ep-coreml` | CoreML | Apple GPU / ANE | ✅ yes |
| `ep-directml` | DirectML | **Intel / AMD / NVIDIA GPU on Windows** | ❌ Windows only |
| `ep-openvino` | OpenVINO | **Intel CPU / iGPU / NPU** | ❌ needs OpenVINO toolkit |
| `ep-cuda` | CUDA | NVIDIA GPU | ❌ needs CUDA toolkit |
| `ep-nnapi` | NNAPI | **Android GPU / NPU** | ❌ needs Android NDK |

EP registration is **strict** (`error_on_failure`): if the EP fails to register,
the run aborts instead of silently falling back to CPU. Note this only catches
registration *failures* — to be certain the EP is in your ORT binary at all,
confirm with `ep_probe` (above), since the prebuilt may or may not bundle a given
EP (see per-EP notes below).

> **Heads-up on `download-binaries`:** the cargo `ep-*` feature enables the ORT
> *Rust* bindings for that EP, but it does **not** change which prebuilt ORT
> library is downloaded — the pyke prebuilt only varies by CPU / CUDA / WebGPU.
> XNNPACK and CoreML happen to be in the macOS prebuilt (verified via `ep_probe`).
> DirectML / OpenVINO / NNAPI may **not** be in the stock prebuilt for their
> platform; if `ep_probe` shows the EP unavailable (or a run aborts), build ORT
> from source with that EP, or point `ORT_LIB_LOCATION` at a system ORT that has
> it. Don't assume "feature compiles" = "EP present."

---

## macOS (this machine) — measured

```sh
RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench \
    --features "ep-xnnpack ep-coreml" -- ep
```

Result (Apple M4, see `results/ep-*.json`): **ort-xnnpack is statistically
indistinguishable from ort-cpu** — XNNPACK registers fine but gives no gain on
this small dynamic-shape BERT (ORT's default MLAS CPU kernels already cover it;
XNNPACK can't claim the dynamic-shape subgraph). **ort-coreml** loses on latency
(high variance) but wins large-batch throughput (batch ≥ 16). Net: on macOS,
Candle Metal (worklog 05) still beats every ORT EP — the EP story matters off-Mac.

---

## Windows — DirectML (the Intel/AMD GPU answer)

DirectML runs on **any DirectX-12 GPU**, so it's the one EP that accelerates
Intel integrated GPUs and AMD GPUs — exactly the gap Candle can't fill.

Prerequisites: Windows 10/11, a DX12 GPU (any modern Intel/AMD/NVIDIA), Rust
MSVC toolchain. **First run `ep_probe` to confirm DirectML is in your ORT build**
— the stock `download-binaries` prebuilt may not bundle the DirectML provider
(unverified; we have no Windows box). If it's absent, get an ORT with DirectML
(Microsoft's Windows ORT package ships it) and point `ORT_LIB_LOCATION` at it.

```powershell
cargo run -p embed-ort --example ep_probe --features directml   # check availability/parity
$env:RAYON_NUM_THREADS=1
cargo run --release -p runner --bin bench --features ep-directml -- ep
```

Notes / gotchas:
- The constructor sets `with_memory_pattern(false)` (DirectML requires it).
  Sequential execution is also required but is ORT's default. If you still hit a
  session-creation error, the next thing to try is `.with_parallel_execution(false)`.
- `with_device_id(0)` selects the first adapter. On a laptop with both an Intel
  iGPU and a discrete GPU, change the id to target the iGPU specifically and
  re-run to compare.
- Compare `ort-directml` vs `ort-cpu` in the output: that delta is the real
  "what do Intel/AMD users gain" number for the meeting.

## Linux/Windows — OpenVINO (Intel CPU/iGPU/NPU)

Best Intel-specific acceleration (iGPU + the newer NPUs).

Prerequisites: install the [OpenVINO toolkit](https://docs.openvino.ai) and
source its `setupvars`. **The OpenVINO EP is almost certainly NOT in the stock
prebuilt** — expect to build ORT from source with `--use_openvino` (or supply a
system ORT via `ORT_LIB_LOCATION`); enabling the cargo feature alone is not
enough. Confirm with `ep_probe` before benching.

```sh
# after installing OpenVINO + an ORT built with the OpenVINO EP
RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench \
    --features ep-openvino -- ep
```

Notes:
- Device is selected by `with_device_type("GPU")` in `load_openvino` — change to
  `"CPU"`, `"NPU"`, or `"AUTO"` to target a different Intel unit and re-run.
- OpenVINO does its own graph compile on first load → expect a slower cold start;
  it caches with `with_cache_dir(...)` if you want to add that.

## Linux — CUDA (NVIDIA, sanity baseline)

Not a gap-filler (Candle already has CUDA), but useful as an upper-bound
reference on an NVIDIA box.

Prerequisites: CUDA toolkit + cuDNN matching the ORT build.

```sh
RAYON_NUM_THREADS=1 cargo run --release -p runner --bin bench \
    --features ep-cuda -- ep
```

## Android — NNAPI

Prerequisites: Android NDK, a Rust Android target (e.g.
`aarch64-linux-android`), and an ORT build with NNAPI for that target (the
prebuilt may not cover it — an ORT-from-source / Maven AAR may be needed).

```sh
cargo build --release -p runner --bin bench \
    --target aarch64-linux-android --features ep-nnapi
# push the binary + run on-device via adb; `ep` mode as usual
```

Notes:
- NNAPI quality varies hugely by device/driver; expect to fall back to CPU on
  some SoCs. `with_cpu_only(false)` / `with_fp16(true)` in `load_nnapi` are the
  knobs to experiment with.
- This is the mobile-acceleration question Larry raised — the answer is
  device-dependent, so report a couple of representative phones, not one number.

---

## What to bring back

For each box you can reach, capture the `ep` table (it writes
`results/ep-<ts>-<host>.json`) and note the **EP-vs-cpu delta**. The decision
question (worklog 06/10) is: *do Intel/AMD/Android users get real acceleration
through ORT where Candle gives them only CPU?* DirectML and OpenVINO deltas on a
Windows/Intel box are the highest-value evidence.
