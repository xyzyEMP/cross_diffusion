> 历史资料：正文保留归档前内容，路径、命令、指标和完成状态不作为当前保证。当前说明见[文档索引](../README.md)。

# Project Status

Updated: 2026-09-29.

## Implemented

- Fixed-arc 8 m / 80-point trajectory representation with validity masks.
- Episode-disjoint ANYmal split and exact nested target-data budgets.
- Frozen validation/test navigation tasks and closed-loop evaluation.
- Navigation-metric checkpoint selection and early stopping.
- Four transfer baselines and an embodiment-conditioned score-decomposition wrapper.
- Platform ID plus 10-dimensional ability encoding.
- Tensor-level `L_diff`, `L_inv`, `L_swap`, residual regularization, adversarial-loss input, and gradient reversal.
- Guards that prohibit paired auxiliary losses on recorded unpaired trajectories.
- Proxy short training with checkpoint/RNG save-and-reload verification.
- Engineering four-way target-platform sampling.
- Real TartanGround local-trajectory pair mining.

## Latest result

One-seed, same-map unseen-episode `pretrain_finetune` results on the frozen 63-task test set:

| Budget | 1% | 10% | 20% | 50% | 100% |
|---:|---:|---:|---:|---:|---:|
| Episode-macro SPL | 0.3861 | 0.4242 | 0.5357 | 0.5071 | **0.6232** |
| Episode-macro SR | 0.3894 | 0.4480 | 0.5636 | 0.5306 | **0.6798** |

Navigation-based validation removed the earlier strong low-budget inversion. The curve is not strictly monotonic: 20% remains slightly above 50%. These are not multi-seed or unseen-map claims.

## Experiment 2/3 boundary

Implemented as engineering interfaces:

- shared/embodiment score decomposition;
- paired masks and proxy objectives;
- invariance and cross-residual swap loss interfaces;
- gradient reversal;
- paired-loss forward/backward smoke tests;
- short proxy training and same-context four-way target-platform sampling on an earlier synthetic fixture set.

Not yet complete:

- conversion of the new recorded matched pairs into common-frame 8 m / 80-point tensors;
- trajectory-pair-grouped train/validation splits;
- an identity classifier trained through the adversarial separation loss;
- non-trivial dual-context intervention/swap;
- full `L_diff/L_inv/L_swap/L_sep` ablations and multi-seed evaluation;
- causal or physical-feasibility conclusions.

The current `L_sep` accepts an external adversarial loss but does not implement the complete classifier-training pipeline. A same-context invariant term may be structurally zero and must not be reported as learned disentanglement.

## Matched-pair status

The current miner produced 46 main ANYmal-Diff candidates at 2.5 m stride. Under the existing ANYmal split, 30 come from training episodes, 16 from held-out test episodes, and none from validation episodes. They must not be randomly split by local window.

The next step is to reconstruct indexed segments, define a shared local frame and task, resample them to 8 m / 80 points, and split by complete trajectory pair while preserving held-out episodes.

## Claim boundary

The repository supports engineering and same-map transfer experiments. It does not yet demonstrate causal embodiment disentanglement, physically valid car-to-quadruped control transfer, or real-world robot performance.
