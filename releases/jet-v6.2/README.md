---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
language:
- en
tags:
- decision-model
- qwen3_5
- transformers
---

# Jet-4B

**v6.2.0 · released 2026-09-26 · focused continuation, selected step 250.**
Jet-4B returns typed decisions and probabilities from supplied options without generating free-form answers.
This is the **full merged BF16 model**, not an adapter, fine-tuned from Qwen3.5-4B. No separate
base-model download is required. It replaces v6.1 in `michaljach/jet-4b`; previous
releases remain accessible by their version tags.

## Run

Linux + NVIDIA CUDA, Python 3.12. Verified with PyTorch 2.11.0+cu128,
Transformers 5.17.0 and flash-linear-attention 0.5.2.

```sh
hf download michaljach/jet-4b --revision v6.2.0 --local-dir jet-4b
cd jet-4b
python -m pip install -r requirements.txt
python jet.py <<'JSON'
{"state":"I was charged twice this month.","questions":{"topic":{"type":"choice","instructions":"What is the primary issue?","criteria":{"billing":"billing or payment problem","bug":"the product is broken"}}}}
JSON
```

```python
from jet import Jet
model = Jet()
result = model.decide("The item arrived broken.", {
    "damaged": {"type": "noul", "instructions": "Is the item damaged?"}
})
```

| Type | Input | Output |
|---|---|---|
| `choice` | 2–255 named options | Selected key and probabilities |
| `score` | 2–10 ordered levels | Expected zero-based score, selected level, probabilities |
| `noul` | Yes/no question | Probability of yes |

Use the included native prompt and restricted label-token readout. A generic
text-generation pipeline does not implement this API. Questions are processed
separately. **The runtime now accepts complete prompts up to 16,384 tokens**;
inputs and options are never silently truncated. The longest reconstructed
API-Bank input (11,495 tokens) passed the standalone runtime check, but full
API-Bank accuracy has not been measured for this release. Context feasibility
does not establish long-context quality across tasks.

FLA is enabled; convolution uses the PyTorch fallback. `confidence` is normalized
inverse entropy, not a calibrated correctness probability. Temperatures are
inherited from v6.1 and were not refitted or established as calibrated for v6.2.

## Training and selection

The parent is full merged Jet-4B v6.1 at
`f446b82727be57da348bb46eccf211414294ab3e`. Parent shard hashes and the exact
adapter hash are in `merge-provenance.json`. A fresh rank/alpha-16 correction LoRA
was trained on that parent; the model was not reset to the original Qwen weights.

- 4,000 examples: 800 banking intent, 800 entity-specific financial sentiment,
  400 sarcasm/literal, and 2,000 broad retention examples.
- BF16 backbone / FP32 adapter, dropout .05, microbatch 1, accumulation 4,
  seed 260925; two trials of 1,000 updates at peak learning rates 3e-6 and 8e-6.
- Validation-selected winner: **3e-6 at step 250**. The higher-rate trial failed
  the selection guards and is not included in this release.
- Selection used 915 examples and an objective of 70% mean focus metrics plus
  30% retention accuracy; every family must stay within two percentage points
  of its initial metric. The winner was frozen before final holdout evaluation.
- Banking uses the training partition of BANKING77, financial supervision uses
  SEntFiN, and sarcasm uses author-labeled iSarcasm training examples.
  Financial headlines and tweet/rephrase groups stay in one split.

## Full-model holdout results

The results below were re-measured on the **full merged model**, after BF16
rounding. Focus holdouts exclude the previous local input corpus listed in the
data audit; retention is reused. Small sample sizes mean these modest differences
do not establish statistical significance or broad benchmark improvement.

| Holdout / metric | Cases | v6.1 | v6.2 merged |
|---|---:|---:|---:|
| Banking / accuracy | 154 | 74.68% | 74.68% |
| Entity sentiment / macro-F1 | 180 | 71.16% | 71.21% |
| Sarcasm / positive-class F1 | 80 | 46.81% | 50.00% |
| Retention / accuracy | 500 | 93.40% | 93.40% |

Financial sentiment is a SEntFiN transfer holdout, **not the FinEntity benchmark**.
Banking and sarcasm rows above are local source holdouts, not their public test
benchmark scores. Exact content/group exclusions do not establish semantic or
pretraining decontamination. Benchmarks informed training focus.

**Decision Index 0.3 (official, 2026-10-07): 40.01, rank 52 of 113.** Run by the
index maintainers on the full suite (110,201 requests, 42 benchmarks; 243 requests
exceed the 16,384-token prompt limit): public 42.17, same skills 40.93, new domains
34.46. Within the index's tie band of the other entries on the same 4B base. Strongest in
Tools & Automation (62.9), weakest in Knowledge & Reasoning (25.4); 66.0% accurate at
80.1% mean confidence (ECE 0.141). [Details](https://quaedra.com/jet-4b/docs/decision-index) ·
[Leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index)

The earlier 25-benchmark comparison belongs to the v6.1 candidate and must not be
attributed to this release.

## Merge verification

FP32 B@A corrections were added to the already merged parent and rounded to BF16.
All 426 tensors loaded without missing, unexpected or mismatched keys; 248 modules
received updates. Nine shards contain 8,411,510,272 parameter bytes.

On 157 fixed verification cases, **0 selected answers changed** between the adapter and merged model. The largest probability difference was **4.76 percentage points**.

These checks include choice, score, noul and a long API-Bank input. The full
914-case holdout passed the predeclared release guards. See evaluation.json,
merge-validation.json and release-manifest.json. Merge equivalence is approximate.

[Website](https://quaedra.com/jet-4b) ·
[Docs](https://quaedra.com/jet-4b/docs/) ·
[Source and experiment records](https://github.com/quaedra/jet-4b) ·
[Training protocol](https://github.com/quaedra/jet-4b/blob/main/experiments/jet-focused-20260925/protocol.md) ·
[Release validation](https://github.com/quaedra/jet-4b/blob/main/experiments/jet-release-20260926/protocol.md)

Apache-2.0 model/runtime; source datasets retain their own licenses. Dataset rows
are not redistributed here. Quality outside the evaluated domains is not established.
