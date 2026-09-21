# Diffusion-Planner on TartanGround

This directory contains a reproducible, zero-shot open-loop transfer benchmark for the released nuPlan Diffusion-Planner checkpoint.

## Code layout

- `config.py`: evaluation and robot-specific limits.
- `data/`: download, trajectory IO, nuPlan feature conversion, terrain, semantics.
- `planning/`: checkpoint runner, basic adapter, diffusion guidance, safety adapter.
- `evaluation/`: open-loop metrics and end-to-end evaluation orchestration.
- `viz/`: static plots and semantic-map rollout GIF generation.
- `run_evaluation.sh`: default reproducible entry point.
- `tests/`: focused transformation, feature, metric, terrain, and adapter tests.

## Protocol

- Input trajectories: `pose_lcam_front.txt` at the documented 10 Hz rate.
- History: 2 seconds (20 frames).
- Prediction: 8 seconds (80 frames).
- Conditioning: the future GT path is converted into a sparse vector-map corridor (`oracle-route`).
- Missing TartanGround annotations: dynamic agents and object tracks are zero padded.
- Output: local-frame `(x, y, yaw)` trajectories.

The default weight-free adapter preserves the checkpoint's spatial path but
retimes it from the last 0.5 seconds of observed robot motion. It enforces a
continuous first step plus robot-specific speed, acceleration, and yaw-rate
limits. Both the untouched `prediction_raw` and adapted `prediction` are saved,
and reports compare raw checkpoint, adapted zero-shot, and constant velocity.

With terrain metrics enabled (the default), a second terrain-aware stage adds:

- differentiable semantic-PCD risk guidance inside DPM-Solver denoising;
- route/terrain candidate reranking across guided and unguided samples;
- Savitzky-Golay smoothing and differential-drive nonholonomic projection;
- terrain look-ahead speed reduction and acceleration-limited safety stopping;
- a final discrete speed, acceleration, yaw-rate, and curvature projection.

The output separates `raw_model_*`, `basic_model_*`, and final `model_*`
metrics so terrain safety gains are not confused with raw checkpoint quality.

This protocol tests whether the released checkpoint can generate a plausible trajectory when given the correct high-level route. It is an upper-bound transfer test, not a non-oracle autonomous-navigation benchmark.

Terrain extraction uses the highest visible semantic-PCD surface in each 0.5 m XY cell and fills only holes within 1 m. This avoids assuming that camera-pose z and global-PCD z share a platform-independent offset (they do not in the released data). Platform thresholds are documented in `config.py` and should be calibrated for a specific physical robot before safety claims.

## Run

The current `town_focused` profile keeps all available trajectories from
ModernCityDowntown and OldTownFall as primary data, plus one representative
trajectory per platform in three auxiliary environments. It contains 46
trajectories (37 primary), covers `omni`, `diff`, and `anymal`, and occupies
approximately 7.52 GiB after verified source ZIP removal:

```bash
conda activate diffusion_planner
python -m tartan.data.download_subset
```

The exact downloaded archive list and extracted byte count are retained in
`TartanGround/subset_manifest.json`. Source ZIPs are reproducible from the
Hugging Face dataset and are removed only after extracted outputs pass checks.

```bash
conda activate diffusion_planner
bash tartan/run_evaluation.sh
```

For the stronger town-focused protocol (552-sample open loop, route-map
quality stratification, matched perturbations, one-second receding-horizon
replanning, kinematic execution feedback, constant-velocity baseline, and
trajectory-cluster bootstrap confidence intervals):

```bash
conda activate diffusion_planner
bash tartan/run_town_focused_v2.sh
```

Its results are written to `outputs/town_focused_v2/`; the previous
`outputs/final/` result is intentionally preserved.

Useful overrides:

```bash
DEVICE=cuda SAMPLES_PER_TRAJECTORY=20 BATCH_SIZE=4 \
OUTPUT_DIR="$PWD/tartan/outputs/gpu_20" \
bash tartan/run_evaluation.sh
```

On a CPU-only machine, start with one sample per trajectory:

```bash
DEVICE=cpu SAMPLES_PER_TRAJECTORY=1 bash tartan/run_evaluation.sh
```

## Outputs

The default final result directory is `tartan/outputs/final`. See
`tartan/outputs/README.md` for the recommended reading order.

- `config.json`: resolved evaluation configuration.
- `run_manifest.json`: environment and checkpoint provenance.
- `dataset_manifest.json`: evaluated environments, robots, trajectory lengths, and semantic-PCD provenance.
- `per_sample_metrics.csv`: one row per evaluated frame.
- `summary.json`: aggregate statistics and protocol notes.
- `summary_by_robot.csv`: omni/diff/anymal comparison.
- `summary_by_environment.csv`: environment comparison.
- `summary_by_robot_environment.csv`: joint breakdown.
- `predictions/*.npz`: history, GT, model prediction, and constant-velocity baseline.
- `visualizations/*.png`: qualitative plots.

Metrics include ADE/FDE, heading error, route deviation, goal progress, path-length ratio, velocity/acceleration/yaw-rate/curvature limits, reverse-motion fraction, and a constant-velocity baseline.

The default run draws three deterministic diffusion samples per input. Metrics are averaged across draws, while `stochastic_spread_mean_m` and `stochastic_endpoint_spread_m` quantify sampling uncertainty. Every draw is retained as `prediction_ensemble` in the sample NPZ.

When semantic PCD files are present, the evaluator also builds cached pose-guided elevation maps and reports platform-specific slope, step-height, roughness, footprint support, and geometric contact-feasibility metrics. These are geometric proxies; they do not replace quadruped foothold planning or contact-dynamics simulation.
