# Decision Index evaluation

Jet-4B is evaluated with the official [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
runner (`decision_index`, pinned to commit `52a698928a9ae5bdf16b75687c903871db29c6e5`
in the `benchmark` extra).

## Official result: Decision Index 0.3

Jet-4B v6.2 (listed as Jet v6.2) is on the [Decision Index 0.3](https://huggingface.co/spaces/multimodalart/jev-decision-index)
leaderboard (edition generated 2026-10-07), run by the index maintainers on the full suite:
110,201 requests across 42 benchmarks, of which Jet-4B answered 109,958. The other 243
exceed its 16,384-token prompt limit and count as unanswered.

| | Score | Rank |
|---|---:|---:|
| **Decision Index 0.3** | **40.01** | 52 of 113 |
| Public benchmarks (20%) | 42.17 | 43 |
| Same skills (50%) | 40.93 | |
| New domains (30%) | 34.46 | |

The index weights the three parts 20/50/30 after equating each across models, so it isn't
a plain weighted sum of the part scores above. Scores within the index's 0.9-point tie band
are tied; Jet-4B's is. Nearby entries on the same
4B base: Hopper (G) 1.2 42.61, JPT-4B 41.63, Kev 4B r10 39.50,
InternLM Intern-Decision 38.21.

| Area | Skill score |
|---|---:|
| Tools & Automation | 62.9 |
| Retrieval & Classification | 48.2 |
| Language Understanding | 44.2 |
| Arts & Human Taste | 30.2 |
| Knowledge & Reasoning | 25.4 |

Calibration on the index's scored cases: 66.0% accurate at 80.1% mean confidence
(ECE 0.141), so Jet-4B is overconfident, most of all in Knowledge & Reasoning and Arts.
New domains is the weakest of the three parts.

The sections below cover running the index yourself. Self-run results are sampled or
partial diagnostics, not official scores.

## Engines

| Engine | Model | Backend |
|---|---|---|
| `decision_index_torch:TorchJetEngine` | Jet-4B: `quaedra/jet` merged weights, or `Qwen/Qwen3.5-4B` plus an adapter | PyTorch, CUDA |
| `decision_index_engine:JetEngine` | Jet 0.6B releases (v5 and earlier), default `quaedra/jet` revision `25ccbd9e` (V5) | MLX (Metal, or CUDA on Linux) |
| `decision_index_ensemble:TwoOrderJetEngine` | as `JetEngine`, averaging original and reversed option order | MLX |

All engines share one request policy (`decision_index_engine.prepare_request`):
complete prompts or an explicit `Unsupported`, no truncation or option pruning,
exact option keys, and full-precision probabilities, with the chosen option's
probability as confidence. The default complete-prompt limit is 8,192 tokens.
Provenance records the resolved model revision, adapter and calibration hashes,
and code hashes.

`jet-bench-index prepare`, `audit` and `report` build compatibility cases and
complete-group diagnostic samples, audit tokenizer coverage, and report official
native metrics on a diagnostic sample. Diagnostic reports never produce an overall index.

## Running

```sh
# Jet-4B on CUDA
uv sync --extra benchmark --inexact
uv run --no-sync python -m decision_index run --engine decision_index_torch:TorchJetEngine \
  --rows artifacts/decision-index/diagnostic/compatibility.jsonl.gz \
  --out artifacts/decision-index/runs/compatibility-v6 \
  --option model=quaedra/jet --option revision=e5b8f610ddb92ffaba596ae452bed32a9fef49ca \
  --option max_tokens=8192

# Qwen3-0.6B releases with MLX on CUDA (the launcher sets up CUDA headers)
bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
  --rows artifacts/decision-index/diagnostic/compatibility.jsonl.gz \
  --out artifacts/decision-index/runs/compatibility-0.6b --option max_tokens=8192
```

Rebuild the public diagnostic corpus, then sample it:

```sh
uv run --extra benchmark python scripts/rebuild_index_diagnostic.py
uv run --extra benchmark jet-bench-index prepare \
  --suite-dir artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt \
  --out artifacts/decision-index/diagnostic --n 500 --allow-partial
```

The official runner resumes from existing results. Keep model revision, capacity
and code fixed within one output directory, and use a new directory per backend.
Run the compatibility file before a full run. Use only the official scorer, and keep
complete source groups, exclusions, full denominators, and track and macro weights.

## Earlier diagnostics

- **Jet-4B v6:** 1,900 sampled requests across 15 datasets, with zero
  errors or unsupported requests. The comparison with published entrants is in
  [`experiments/jet-4b-evaluation/results.md`](../experiments/jet-4b-evaluation/results.md).
  The full-benchmark expansion and the remaining blockers to an overall score are in
  [`experiments/jet-4b-full-index/`](../experiments/jet-4b-full-index/README.md).
- **Jet 0.6B (jet-1, 2026-09-23):** a partial public rebuild of 23,113 requests
  over eight benchmarks, and a compatibility pass (26/26). Coverage is in
  `docs/bench/decision-index/coverage.json`; `diagnostic-8k.json` is an incomplete
  snapshot of a run that was stopped early.

## Blockers to a self-run overall score

- `multimodalart/decision-index-suite` returns HTTP 404
  ([apolinario/decision-index#1](https://github.com/apolinario/decision-index/issues/1)).
- HLE (`cais/hle`) needs gated-dataset access.
- The public reproduction kit lacks release-v2 recipes for several panel additions.
  Without the matching corpus, any subset average is not comparable to the index.
