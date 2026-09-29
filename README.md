# Cross-Diffusion

Cross-Diffusion studies transfer of a nuPlan-pretrained diffusion trajectory planner to TartanGround embodied navigation. The current target is body-level local planning: an 8 m, 80-point SE(2) path, not joint- or motor-level control.

## Current scope

This repository contains:

- the Diffusion Planner backbone and TartanGround preprocessing/evaluation;
- episode-disjoint train/validation/test splits and exact nested data budgets;
- navigation-metric checkpoint selection with early stopping;
- full fine-tuning, frozen-adapter, joint-training, and embodiment-conditioned baselines;
- shared/embodiment score decomposition with per-denoising-step residual correction;
- proxy interfaces for diffusion, invariance, swap, separation, and gradient reversal;
- real local-trajectory pair mining for ANYmal, Diff, and Omni;
- engineering smoke runners for paired losses and four-way target-platform sampling.

The decomposition/swap path is still an engineering prototype. Formal causal disentanglement, a trained adversarial separation classifier, non-trivial dual-context swap, and materialization of the new recorded matched pairs are not complete. See [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Data protocol

| Split | Episodes | Use |
|---|---:|---|
| Train | 16 | 2,048 dense windows |
| Validation | 3 | loss monitoring and 35 frozen navigation tasks |
| Test | 5 | 63 frozen non-overlapping 8 m tasks |

Exact nested training budgets are 20/205/410/1024/2048 windows for 1/10/20/50/100%. Every budget covers all 16 training episodes; validation and test remain unseen.

## Model flow

```text
pose + observable map + fixed goal
  -> ego / lanes / route_lanes
  -> Diffusion Planner shared score
  -> optional embodiment residual at each denoising step
  -> 80-point local SE(2) path
  -> receding-horizon rollout
  -> SR / CR / SPL / goal progress
```

Key code:

- `tartan/research_score/model/score_decomposition.py`
- `tartan/research_score/training/losses.py`
- `tartan/research_score/scripts/train_finetune_navigation_earlystop.py`
- `tartan/research_score/scripts/evaluate_nonoverlap_segments.py`
- `tartan/research_score/scripts/run_proxy_smoke.py`
- `tartan/research_score/scripts/run_proxy_swap.py`
- `pair/run_pair_mining.py`

## Installation

Set up nuPlan following its official documentation, then:

```bash
conda create -n diffusion_planner python=3.9
conda activate diffusion_planner

git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
pip install -r requirements.txt

cd ..
git clone https://github.com/xyzyEMP/cross_diffusion.git
cd cross_diffusion
pip install -e .
pip install -r requirements_torch.txt
```

## Tests

```bash
python -m pytest -q tartan/research_score/tests
python -m unittest discover -s pair -p 'test_*.py' -v
```

Datasets, feature caches, checkpoints, experiment outputs, internal plans, and detailed handoff documents are intentionally excluded from Git.
