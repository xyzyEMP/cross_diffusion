## Getting Started

- Setup the nuPlan dataset following the [official documentation](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html).
- Setup conda environment.

```bash
conda create -n diffusion_planner python=3.9
conda activate diffusion_planner

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r requirements.txt

# setup cross_diffusion
cd ..
git clone https://github.com/xyzyEMP/cross_diffusion.git && cd cross_diffusion
pip install -e .
pip install -r requirements_torch.txt
```

## Current cross-embodiment comparison

This repository studies transfer from the nuPlan Car diffusion planner to TartanGround ANYmal. The current comparison contains four methods:

- `pretrain_finetune`: load the Car checkpoint and fine-tune all parameters on ANYmal;
- `pretrain_adapter`: freeze the Car backbone and train only the embodiment adapter;
- `joint_train`: alternate Car and ANYmal batches while updating one backbone;
- `emb_cond_diffusion`: joint training with platform ID, ability vector, and a residual correction at every denoising step.

The formal protocol uses full-episode train/validation/test split `16/3/5`. Training uses dense 8 m / 80-point windows; held-out closed-loop testing uses 63 non-overlapping 8 m tasks from the five test episodes.

## Reproduction entry points

Set the dataset, checkpoint, and output paths in your environment before launching:

```bash
export PROJECT_ROOT=/path/to/cross_diffusion
export PYTHON_BIN=python
export SOURCE_ARGS=$PROJECT_ROOT/checkpoints/args.json
export SOURCE_CKPT=/path/to/model.pth
export OUTPUT_ROOT=/path/to/experiment_outputs
cd "$PROJECT_ROOT"

# Train the four-method, three-budget formal matrix.
bash tartan/research_score/scripts/run_stage07_formal_seed11_v3.sh

# Evaluate the selected checkpoints with non-overlapping test segments.
bash tartan/research_score/scripts/run_stage07_nonoverlap_eval.sh
```

Large checkpoints, datasets, feature caches, and experiment outputs are intentionally excluded from Git.
