# 实验一运行手册

本文件用于复现 `nuPlan Car → TartanGround ANYmal` 的正式迁移实验。正式比较包含四种预训练迁移方法、三种目标域训练数据预算；Proxy 不进入实验一主表。

## 1. 环境与路径

```bash
ssh -p 56137 root@220.196.169.30
conda activate diffusion-planner
cd /zeron-vepfs/tjqc/cross-diffusion

export PROJECT_ROOT=/zeron-vepfs/tjqc/cross-diffusion
export OUTPUT_ROOT=/tj-share/cross_diffusion_workdir/research_score_v1_2
export DATA_RUN=$OUTPUT_ROOT/01_data_audit/stage03_native_v123_remediation_final_20260911T170000Z_nogit/transfer_primary
```

四种方法：

- `pretrain_finetune`：nuPlan checkpoint 初始化，使用 ANYmal 全参数微调；
- `pretrain_adapter`：冻结 nuPlan 主干，仅训练轻量具身适配模块；
- `joint_train`：nuPlan 初始化，交替使用 Car 和 ANYmal batch；
- `emb_cond_diffusion`：在联合训练基础上加入平台 ID 和能力条件。

名义预算为 1%/10%/100%。由于按完整 episode 划分且禁止拆分，当前实际为 128/256/2048 个 ANYmal 训练窗口。验证集和测试 episode 对所有方法固定。

## 2. 一次性构建训练特征缓存

缓存只消除每个训练进程重复执行 A* 的开销，不改变样本或输入。

```bash
mkdir -p $OUTPUT_ROOT/02_feature_cache

$CONDA_PREFIX/bin/python -m tartan.research_score.scripts.materialize_target_features \
  --manifest $DATA_RUN/target_train_windows.jsonl \
  --args checkpoints/args.json \
  --output $OUTPUT_ROOT/02_feature_cache/target_train_features.pt

$CONDA_PREFIX/bin/python -m tartan.research_score.scripts.materialize_target_features \
  --manifest $DATA_RUN/target_val_windows.jsonl \
  --args checkpoints/args.json \
  --output $OUTPUT_ROOT/02_feature_cache/target_val_features.pt

$CONDA_PREFIX/bin/python -m tartan.research_score.scripts.materialize_source_features \
  --manifest $DATA_RUN/source_length_audit.jsonl --samples 4096 --seed 20260914 \
  --output $OUTPUT_ROOT/02_feature_cache/source_car_features_4096.pt
```

缓存已经存在时不要重复执行。

## 3. 单任务正式训练命令

下面示例训练 `pretrain_finetune / 10% / seed 11`：

```bash
CUDA_VISIBLE_DEVICES=0 $CONDA_PREFIX/bin/python -u \
  -m tartan.research_score.scripts.train_method_formal \
  --method pretrain_finetune \
  --train-cache $OUTPUT_ROOT/02_feature_cache/target_train_features.pt \
  --val-cache $OUTPUT_ROOT/02_feature_cache/target_val_features.pt \
  --source-cache $OUTPUT_ROOT/02_feature_cache/source_car_features_4096.pt \
  --args checkpoints/args.json \
  --checkpoint checkpoints/model.pth \
  --output $OUTPUT_ROOT/05_transfer/manual_pretrain_finetune_10pct_seed11 \
  --budget 10 --max-target-updates 10000 --min-target-updates 5000 \
  --val-every-updates 250 --patience-validations 10 \
  --batch-size 64 --seed 11 --amp
```

训练以目标域 optimizer update 计数，而不是用会随预算改变的 epoch 作为停止尺度。所有方法最多获得 10,000 次 ANYmal 更新；联合方法每个 ANYmal batch 后增加一个 Car batch，因此 ANYmal 更新数仍与其他方法一致。每 250 次 ANYmal 更新在完整的 384 个窗口上验证，至少训练 5,000 次，连续 10 次验证无改善时早停，并在平台期降低学习率。训练保存 `best.pt`、`last.pt` 和 `metrics.json`。

## 4. seed 11 v3 正式训练结果

```bash
# 现有12项保留方法的收敛结果位于：
ls "$OUTPUT_ROOT/05_transfer/formal_seed11_v3"
```

四种方法 × 3 预算的 12 项有效 checkpoint 位于：

```text
$OUTPUT_ROOT/05_transfer/formal_seed11_v3/
```

已有完整 `metrics.json` 的任务会跳过；存在不完整目录时脚本停止并要求人工确认，不会覆盖结果。

## 5. v1.2.4 非重叠8 m 闭环评价

```bash
bash tartan/research_score/scripts/run_stage07_nonoverlap_eval.sh
```

脚本首先从5条 held-out episode构建63个非重叠8 m测试任务，再评价全12个 checkpoint。结果位于 `$OUTPUT_ROOT/05_transfer/formal_seed11_v3_eval_nonoverlap8m/`。每项包含 `per_segment.csv` 和 `summary.json`，核心指标为：

- `sr`：成功到达固定目标的 episode 比例；
- `cr`：发生碰撞的 episode 比例；
- `spl`：成功率与路径效率的联合指标；
- `stuck_rate`、`route_failure_rate`、`goal_progress`；
- `mean_inference_s`：平均单次重规划推理耗时。

主表使用63个段级任务的 SR/CR/SPL，并同时报告5条 episode的等权宏平均。段不重叠，但不将同一 episode 内的段声称为完全独立场景。

## 6. 第一轮筛查与正式结果的区别

`seed11_first_round/` 和旧的5-episode整段评价只作为历史管线证据。v1.2.4 主结果必须使用 `formal_seed11_v3/` 的有效 checkpoint 和 `formal_seed11_v3_eval_nonoverlap8m/` 的段级及 episode-macro 指标。
