#!/usr/bin/env bash
# Secondary metrics: cold start, peak RSS, binary size, clean build time.
# Run pinned + on AC, from the repo root:
#   RAYON_NUM_THREADS=1 scripts/secondary.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SAMPLES=7
now() { python3 -c 'import time; print(time.time())'; }

echo "== building single-engine binaries (release) =="
cargo build --release --bin single_candle --bin single_burn >/dev/null 2>&1

measure_runtime() {  # $1=binary  -> appends "coldms rssbytes" lines to stdout
    local bin="$1"
    for _ in $(seq "$SAMPLES"); do
        local err cold rss
        err="$(mktemp)"
        cold="$(/usr/bin/time -l "./target/release/$bin" 2>"$err")"
        rss="$(grep 'maximum resident set size' "$err" | awk '{print $1}')"
        rm -f "$err"
        echo "$cold $rss"
    done
}

echo "== cold start + peak RSS ($SAMPLES runs each) =="
measure_runtime single_candle > /tmp/cvb_candle.txt
measure_runtime single_burn   > /tmp/cvb_burn.txt

echo "== binary size (stripped) =="
strip -o /tmp/cvb_sc ./target/release/single_candle
strip -o /tmp/cvb_sb ./target/release/single_burn
CANDLE_BIN=$(stat -f%z /tmp/cvb_sc)
BURN_BIN=$(stat -f%z /tmp/cvb_sb)

echo "== clean build time (full clean before each; shared deps counted in both) =="
cargo clean >/dev/null 2>&1
t0=$(now); cargo build --release --bin single_candle >/dev/null 2>&1; t1=$(now)
CANDLE_BUILD=$(python3 -c "print(f'{$t1-$t0:.1f}')")
cargo clean >/dev/null 2>&1
t0=$(now); cargo build --release --bin single_burn >/dev/null 2>&1; t1=$(now)
BURN_BUILD=$(python3 -c "print(f'{$t1-$t0:.1f}')")

echo "== aggregate -> JSON + summary =="
python3 - "$CANDLE_BIN" "$BURN_BIN" "$CANDLE_BUILD" "$BURN_BUILD" <<'PY'
import sys, json, glob, statistics as st, subprocess, os

candle_bin, burn_bin, candle_build, burn_build = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def load(path):
    cold, rss = [], []
    for line in open(path):
        c, r = line.split()
        cold.append(float(c)); rss.append(float(r))
    return cold, rss

cc, cr = load("/tmp/cvb_candle.txt")
bc, br = load("/tmp/cvb_burn.txt")

def cmd(*a):
    try: return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except Exception: return ""

mb = lambda b: round(b/1_000_000, 1)
report = {
  "environment": {
    "cpu": cmd("sysctl","-n","machdep.cpu.brand_string"),
    "rustc": cmd("rustc","--version"),
    "rayon_threads": os.environ.get("RAYON_NUM_THREADS","unset"),
    "on_ac_power": "AC Power" in cmd("pmset","-g","batt"),
  },
  "cold_start_ms": {
    "candle-cpu": {"median": round(st.median(cc),1), "min": round(min(cc),1)},
    "burn-ndarray": {"median": round(st.median(bc),1), "min": round(min(bc),1)},
  },
  "peak_rss_mb": {
    "candle-cpu": {"median": mb(st.median(cr))},
    "burn-ndarray": {"median": mb(st.median(br))},
  },
  "binary_size_mb": {
    "candle-cpu": mb(int(candle_bin)),
    "burn-ndarray": mb(int(burn_bin)),
  },
  "clean_build_s": {
    "candle-cpu": float(candle_build),
    "burn-ndarray": float(burn_build),
  },
}
ts = cmd("date","-u","+%Y%m%dT%H%M%SZ") or "run"
host = cmd("hostname","-s") or "host"
os.makedirs("results", exist_ok=True)
path = f"results/secondary-{ts}-{host}.json"
json.dump(report, open(path,"w"), indent=2)

print(f"\n{'metric':<16}{'candle-cpu':>14}{'burn-ndarray':>16}")
print(f"{'cold start (ms)':<16}{report['cold_start_ms']['candle-cpu']['median']:>14}{report['cold_start_ms']['burn-ndarray']['median']:>16}")
print(f"{'peak RSS (MB)':<16}{report['peak_rss_mb']['candle-cpu']['median']:>14}{report['peak_rss_mb']['burn-ndarray']['median']:>16}")
print(f"{'binary (MB)':<16}{report['binary_size_mb']['candle-cpu']:>14}{report['binary_size_mb']['burn-ndarray']:>16}")
print(f"{'clean build (s)':<16}{report['clean_build_s']['candle-cpu']:>14}{report['clean_build_s']['burn-ndarray']:>16}")
print(f"\nwrote {path}")
PY
