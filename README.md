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

Run commands are added here as each stage of the pipeline lands.

Datasets, model outputs and hidden states are written to `data/` and `outputs/`, which are
not committed.
