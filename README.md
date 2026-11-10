# Answer or Abstain? Do Small LLMs Follow Confidence Targets?

CS 7650 (Fall 2026) course project, Georgia Tech.

## Overview

Under a confidence-target scoring rule (Kalai et al., 2025), a correct answer earns 1 point,
"I don't know" earns 0, and a wrong answer costs t/(1-t) points, so answering only pays off
when the model is more than t confident. We test whether small open LLMs follow this rule
on multiple-choice questions, and whether signals inside the model can implement it when
prompting alone does not.

We compare abstention policies:

- **P1, prompt-only:** the rule is stated and the model chooses an answer or "I don't know".
- **P2, verbalized confidence:** the model states a confidence and we threshold it at t.
- **P3 / P3b, answer probability:** we threshold the top answer-letter probability, raw and
  after temperature scaling.
- **P4, probe:** a logistic-regression probe on hidden states predicts correctness.

Each policy is scored at t in {0, 0.5, 0.75, 0.9} against always-answer, random-abstention
and oracle baselines, using penalized score, abstention rate, selective accuracy and regret.

Models: Qwen3-1.7B, 4B, 8B and Llama-3.1-8B-Instruct. Data: MMLU-Pro (primary) and GPQA
Diamond (out-of-distribution).

## Setup

```bash
pip install -r requirements.txt
```

A Hugging Face account is needed. Llama-3.1-8B-Instruct and GPQA Diamond are gated, so accept
their terms on the Hub and log in:

```bash
huggingface-cli login
```

## How to run

Inference needs a CUDA GPU. The easiest route is `notebooks/colab_runs.ipynb` in Google
Colab, which runs every step below in order. From a shell:

```bash
# Unit tests (CPU only)
python -m pytest -q

# Neutral prompt: baseline answers, answer-letter logits (P3) and hidden states (P4)
python -m src.infer --model Qwen/Qwen3-1.7B --prompt neutral --save-hidden

# P1: one pass per confidence target
for t in 0 0.5 0.75 0.9; do python -m src.infer --model Qwen/Qwen3-1.7B --prompt p1 --t $t; done

# P2: verbalized confidence (optional)
python -m src.generate --model Qwen/Qwen3-1.7B

# Fit P3b and P4, score every policy on the test split, write tables and figures
python -m src.report --run-dir outputs/Qwen__Qwen3-1.7B
```

Add `--limit 200` to an inference command for a quick pilot. Runs are resumable: rerunning a
command skips chunks that are already written.

Datasets, model outputs, hidden states and reports are written to `data/` and `outputs/`,
which are not committed.

## Results

`results/` holds a snapshot of the report output for each model run so far, on the MMLU-Pro
test split (2,407 questions):

| File | Contents |
|---|---|
| `results.csv` | One row per policy, confidence target and subject group (`all`, `math_heavy`, `other`): penalized score, abstention rate, selective accuracy, baselines, oracle score and regret |
| `signals.csv` | Per confidence signal: AUROC, AURC, expected calibration error, mean confidence, accuracy |
| `summary.json` | Test accuracy, fitted temperature, selected probe layer, P1 answer-flip rates, P2 parse rate |
| `probe_layers.csv` | Probe validation and train metrics for every layer and regularization strength |
| `*.png` | Abstention rate and penalized score by target, risk-coverage curves, probe AUROC by layer |

These files are copied from `outputs/<model>/report/` after a run; regenerate them with
`python -m src.report` and copy them again if the runs change.
