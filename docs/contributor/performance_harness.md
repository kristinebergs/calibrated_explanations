# Performance Harness

This page describes how to profile and benchmark `calibrated-explanations`.
The repository ships several purpose-built scripts; choose the one that matches
your goal.

---

## Quick-start: micro benchmarks

`scripts/perf/run_micro_benchmarks.py` uses `WrapCalibratedExplainer` (CE-first
pattern) to time `fit`, `calibrate`, and `predict` on both classification and
regression tasks.

```bash
python scripts/perf/run_micro_benchmarks.py --output reports/perf/micro.json --pretty
```

---

## Full baseline snapshot

`scripts/perf/collect_baseline.py` records import time, RSS memory (`psutil`),
`tracemalloc` peak, public API symbol inventory, and runtime benchmarks.  Run
this before and after a change to get a before/after pair.

```bash
python scripts/perf/collect_baseline.py \
    --output tests/benchmarks/baseline_$(date +%Y%m%d).json \
    --pretty
```

Windows PowerShell:

```powershell
python scripts/perf/collect_baseline.py `
    --output tests/benchmarks/baseline_$(Get-Date -Format yyyyMMdd).json `
    --pretty
```

---

## Regression check

`scripts/perf/check_perf_regression.py` compares a saved baseline against
current metrics using the thresholds in `tests/benchmarks/perf_thresholds.json`.

```bash
python scripts/perf/check_perf_regression.py \
    --baseline tests/benchmarks/baseline_YYYYMMDD.json \
    --thresholds tests/benchmarks/perf_thresholds.json
```

Exits non-zero if any threshold is exceeded or if public API symbols were
removed.

---

## Legacy vs modern pipeline comparison

The historical five-strategy comparison and its result report are maintained
in the
[studies repository](https://github.com/kristinebergs/calibrated-explanations-studies/tree/main/legacy/scripts).
Follow the repository's study-specific instructions when reproducing those
numbers. Core contributors should use the maintained harnesses on this page for
current library performance work.

Prerequisites: the same environment used for the rest of the test suite
(`pip install -e .[dev]`).

---

## Streaming serialization benchmark

`scripts/perf/stream_benchmark.py` measures elapsed time and peak memory for
serialising `N` synthetic explanations through the streaming API.

```bash
python scripts/perf/stream_benchmark.py --n 10000 --chunk 256 --format jsonl
```

---

## Committed baselines

Baseline snapshots live in `tests/benchmarks/`.  The thresholds in
`tests/benchmarks/perf_thresholds.json` govern import time and explanation
latency.  Update the baseline file and commit it when a deliberate performance
change lands.

---

## When to run

| Scenario | Recommended script |
|---|---|
| Quick sanity check after a change | `run_micro_benchmarks.py` |
| Before/after comparison for a PR | `collect_baseline.py` + `check_perf_regression.py` |
| Reproducing pipeline speedup numbers | `compare_explain_performance.py` |
| Streaming throughput check | `stream_benchmark.py` |
