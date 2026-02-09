# Rubrics-as-an-Attack-Surface

## Configurations

1. **Requirements**
```bash
    conda create -n rubrics python=3.9
    conda activate rubrics
    pip install -r requirements.txt
```
2. **Set API keys**
- Set the required API keys in the `.env` file before running experiments.


## Data

We use five human-preference datasets (UltraFeedback, ChatbotArena, RMB, Anthropic hh-rlhf, PKU-SafeRLHF) to construct four benchmark–target settings: **Ultra-Real** and **Ultra-Creative** for helpfulness (UltraFeedback → ChatbotArena), and **SafeRLHF–RMB** and **Anthropic–SafeRLHF** for harmlessness. All data is converted to a uniform pairwise preference format; benchmarks enforce rubric preservation, while targets measure deployment-relevant preference drift, with downstream policy experiments on Ultra-Real and Anthropic–SafeRLHF.

### Scripts

The full data pipeline (download, preprocessing, filtering, and domain splitting) is run with:

```bash
sh ./data/scripts/dataset.sh
```

Run from the repository root.

### Directory Structure

After the pipeline completes, the directory layout is:

```text
data/
├── helpfulness/
│   ├── Ultra-Real/
│   │   ├── Ultra-Real-Bench/
│   │   │   ├── train.jsonl
│   │   │   ├── val.jsonl
│   │   │   └── test.jsonl
│   │   └── Ultra-Real-Target/
│   │       ├── train.jsonl
│   │       ├── val.jsonl
│   │       └── test.jsonl
│   └── ...
├── harmlessness/
│   ├── Anthropic-SafeRLHF/
│   │   ├── Anthropic-SafeRLHF-Bench/
│   │   │   ├── train.jsonl
│   │   │   ├── val.jsonl
│   │   │   └── test.jsonl
│   │   └── Anthropic-SafeRLHF-Target/
│   │       ├── train.jsonl
│   │       ├── val.jsonl
│   │       └── test.jsonl
│   └── ...
```
**Bench vs. Target.**  
For each dataset configuration, **Bench** denotes the *benchmark domain* used during rubric development, while **Target** denotes a *held-out deployment domain* used to evaluate generalization and preference drift. Rubric edits are validated exclusively on the Bench domain and never optimized using Target data.

**Data splits and usage.**
- **train.jsonl**: Used for *rubric search and refinement*.
- **val.jsonl**: Used for *rubric selection*, ensuring benchmark compliance.
- **test.jsonl**: Used exclusively for *evaluation of Rubric-Induced Preference Drift (RIPD)* and is never accessed during rubric editing.



## Biased Rubric Search


## Downstream Policy Misalignment Evaluation

### dpo training
run `train_dpo.sh` to train your policy with training data as well as their labels by the given rubrics.

### policy evaluation
run 5 steps at your choice in `eval.sh`, to:
 - generate responses, 
 - score them, 
 - analyze win-rate, 
 - select bon response, 
 - eval by 3rd party judge.
