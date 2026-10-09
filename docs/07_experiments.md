# 实验与资源

本页统一说明训练、评估、数据构建和部署使用的固定路径与 CLI。本文约束新运行；既有运行的实际地址只在对应历史实验记录中保留，不追溯修改。当前代码不会自动创建 cache、split、manifest 或建议的输出子目录。

## 固定连接与路径

服务器使用 `ssh -p 56137 root@220.196.169.30` 登录。命令从项目根目录执行；Python 使用 `PY=/root/miniconda3/envs/diffusion-planner/bin/python3.9`。这是服务器上 conda 环境 `diffusion-planner` 的可执行文件，不要自行切换环境。

| 变量/路径 | 用途 | 固定位置/规则 |
|---|---|---|
| `DATA_ROOT` | 原始数据 | `/tj-share/tartanground/ModularNeighborhood` |
| `INPUT_ROOT` | 已有 cache、manifest 等历史输入，只读引用 | `/tj-share/cross_diffusion_workdir` |
| `SOURCE_ROOT` | 已有 source args 和初始化 checkpoint，只读引用 | `/zeron-vepfs/tjqc/cross-diffusion`；文件位于 `checkpoints/args.json`、`checkpoints/model.pth` |
| `RUNS_ROOT` | 新实验的共同输出父目录 | `/root/cross_workdir` |
| `EXP_ID` | 每次运行唯一标识 | 以 `YYYYMMDD_` 开头，例如 `20261009_anymal_seed11_run01`；已有目录不可复用 |
| `RUN_DIR` | 本次运行目录 | `$RUNS_ROOT/$EXP_ID` |

`EXP_ID` 对应 `$RUNS_ROOT` 的直接子目录，因此只有该层目录名必须以日期开头。`RUN_DIR` 内的 checkpoint、summary、日志等文件及子目录沿用工具的现有命名，不要求加日期前缀。

## 文件落点与操作约束

| 类别 | 位置与规则 |
|---|---|
| 原始数据 | 从 `$DATA_ROOT` 读取；不覆盖、迁移或在此生成实验产物 |
| 已有 cache、manifest | 从 `$INPUT_ROOT` 只读引用。历史输入可以继续引用旧工作区或 archive 中的文件，不迁移或重生成输入 |
| source args、初始化权重 | 从 `$SOURCE_ROOT/checkpoints/` 只读引用，不与新训练 checkpoint 混放 |
| 所有新生成产物 | 日志、报告、图表、checkpoint、评估结果、cache、manifest、pair 结果等均保存在所属 `$RUN_DIR` 内 |
| 建议的输出子路径 | 可由执行者在 `$RUN_DIR` 下指定 `data/cache/`、`data/manifests/`、`data/pairs/` 等；这些只是路径建议，不表示现有工具会自动创建它们。运行前按入口要求创建目录 |
| 代码快照 | 记录实际代码位置；不得修改正在运行的代码快照或输入，不得搬动运行目录或由其他任务覆盖产物 |
| 历史产物 | 旧工作区、archive 和已有实验输出继续只读引用；未经确认不得移动、覆盖或删除 |
| 临时文件 | 可使用 `/tmp` 中带任务标识的文件；需要保留的运行证据也保存到所属 `$RUN_DIR` |
| 实验文档 | `docs/experiments/<experiment>.md`；记录 run ID、实际代码/输入/输出地址、参数、完成状态和结果。通用路径规则只维护在本页 |

新实验在创建输出前确认 `$RUN_DIR` 不存在；每次运行使用新的日期前缀 ID，不复用已有目录。执行记录实际启动命令、输入/输出地址，并区分训练、验证和测试数据。路径未规定或不可用时先确认，不选择静默替代位置。新规则不要求改写历史文档；例如 [Experiment 1](experiments/experiment1.md) 保留该次运行的历史地址。

## 已有输入示例

下列 cache、验证 manifest 和测试 manifest 是已有输入，示例只读引用它们。测试 manifest 的历史位置及其实际记录见 [Experiment 1](experiments/experiment1.md)。新原始数据根 `$DATA_ROOT` 与包含这些已有 cache/manifest 的 `$INPUT_ROOT` 是不同用途，不应混用。manifest 内引用的地图文件也必须可读。

## CLI 参数

参数名与当前脚本 argparse 一致；训练、评估不会构建 cache、split 或 manifest。

| 入口 | 参数 | 输入/输出 |
|---|---|---|
| `scripts/train.py` | `--train-cache`, `--val-cache` | 训练和验证 cache 文件 |
| `scripts/train.py` | `--val-navigation-manifest` | 导航验证 JSONL manifest |
| `scripts/train.py` | `--args`, `--checkpoint` | source args JSON、source checkpoint |
| `scripts/train.py` | `--output` | 训练输出目录 |
| `scripts/evaluate.py` | `--method`, `--checkpoint`, `--args` | 方法名、导航 checkpoint、args JSON |
| `scripts/evaluate.py` | `--test-manifest` | 测试 JSONL manifest |
| `scripts/evaluate.py` | `--output` | 评估输出目录 |
| `datasets/pair/run_pair_mining.py` | `--output-dir` | pair 构建输出目录 |

以下示例从项目根目录执行，复用已存在输入，并把新产物归入一个新 run。示例的参数仅展示路径组织，不规定未来实验的超参数或数据划分。

```bash
PY=/root/miniconda3/envs/diffusion-planner/bin/python3.9
DATA_ROOT=/tj-share/tartanground/ModularNeighborhood
INPUT_ROOT=/tj-share/cross_diffusion_workdir
SOURCE_ROOT=/zeron-vepfs/tjqc/cross-diffusion
RUNS_ROOT=/root/cross_workdir
EXP_ID=20261009_anymal_seed11_run01
RUN_DIR="$RUNS_ROOT/$EXP_ID"

set -e
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/train" "$RUN_DIR/eval"

"$PY" -m scripts.train \
  --train-cache "$INPUT_ROOT/cache/target_train_features_d029_extended.pt" \
  --val-cache "$INPUT_ROOT/cache/target_val_features.pt" \
  --val-navigation-manifest "$INPUT_ROOT/data/navigation/val.jsonl" \
  --args "$SOURCE_ROOT/checkpoints/args.json" \
  --checkpoint "$SOURCE_ROOT/checkpoints/model.pth" \
  --output "$RUN_DIR/train/pretrain_finetune_100pct_seed11" \
  --budget 100 --batch-size 64 --seed 11 --max-epochs 100 --min-epochs 30 \
  --navigation-every-epochs 5 --scheduler-every-epochs 10 \
  --early-stop-patience 8 --inference-seeds 11 23 47 --amp

"$PY" -m scripts.evaluate \
  --method pretrain_finetune \
  --checkpoint "$RUN_DIR/train/pretrain_finetune_100pct_seed11/navigation_best.pt" \
  --args "$SOURCE_ROOT/checkpoints/args.json" \
  --test-manifest "$INPUT_ROOT/archive/experiment1/inputs/navigation_test.jsonl" \
  --output "$RUN_DIR/eval/pretrain_finetune_100pct_seed11" \
  --inference-seed 10000
```

两个入口以 `exist_ok=False` 创建 `--output` 路径，需保证对应路径不存在；上述示例仅检查 `RUN_DIR`。实际运行的参数、输入核验、结果及状态记录在对应的 `docs/experiments/<experiment>.md`。
