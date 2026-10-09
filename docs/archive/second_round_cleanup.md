# 第二轮结构整理记录

本轮以第一轮整理后的工作树为基线，不以 Git HEAD 的旧目录为基线。范围是命名、引用、职责归位和历史入口归档；保留模型结构、Forward、Loss、预处理算法、训练协议及超参数。下表仅记录本轮变化。

## 主要目录：修改前

省略未受影响的文件；目录树中的 `...` 表示原有内容继续保留。

```text
cross_diffusion/
├── datasets/
│   ├── core.py / nuplan_dataset.py / paired_dataset.py
│   ├── representation_bridge.py
│   ├── audit.py / split_builder.py / trajectory_spec.py
│   ├── pair/ / transforms/ / ...
├── models/representation_bridge.py / ...
├── losses/research_score.py
├── scripts/
│   ├── build_native_v122.py / validate_native_v122.py
│   ├── train_method_formal.py / train_method_equal_epochs_fine_select.py
│   ├── build_nonoverlap_segments.py / build_nonoverlap8m_test.py
│   ├── build_manifests.py / build_remediation.py
│   ├── diagnose_length_recovery.py / validate_remediation.py
│   ├── audit_routes_remediation.py / ...
├── configs/experiment/
│   ├── base.yaml / stage02.yaml
│   ├── stage03_v1_2.yaml / stage03_v1_2_1.yaml / stage03_v1_2_2.yaml
│   ├── stage03_v1_2_3.yaml / ...
├── experiments/
│   ├── run_stage03.sh / run_stage07_seed11.sh / run_stage07_seed11_retry6.sh
│   └── ...
├── engine/ / evaluation/ / utils/ / tests/
├── docs/README.md / 01_overview.md 至 08_development.md / archive/
└── README.md / setup.py / requirements_torch.txt
```

## 主要目录：修改后

```text
cross_diffusion/
├── datasets/
│   ├── source_contract.py       # source 版本及输入身份契约
│   ├── preprocessing.py         # 现有几何、轨迹及 split 工具
│   ├── trajectory_resampling.py # 数据侧重采样
│   ├── pair/validation.py       # pair 校验；原 pair 模块继续保留
│   ├── archive/                 # 三个原 re-export 模块
│   ├── transforms/ / ...
├── models/source_trajectory_bridge.py / ...
├── losses/objectives.py
├── scripts/
│   ├── build_native_transfer_manifests.py / validate_native_transfer_manifests.py
│   ├── train_transfer_early_stopping.py
│   ├── train_transfer_fixed_updates.py
│   ├── train_finetune_navigation_earlystop.py
│   ├── build_nonoverlap_segments.py
│   ├── archive/                 # 六个旧/重复构建与诊断入口
│   └── ...                     # 其余训练、评估和审计入口保留
├── configs/experiment/
│   ├── stage02.yaml / stage03_v1_2_3.yaml / ...
│   └── archive/                 # 四个旧/重复配置
├── experiments/
│   ├── archive/                 # 三个旧启动脚本
│   └── ...                     # 当前实验 recipe 保留
├── engine/ / evaluation/ / utils/ / tests/
├── docs/
│   ├── README.md / 01_overview.md 至 08_development.md
│   └── archive/                # 六份原历史文档及本整理记录
└── README.md / setup.py / requirements_torch.txt
```

## 重命名与迁移对照

全部 26 个原文件保留：10 个名称或职责调整、16 个归档。新建四个空 `archive/__init__.py` 仅供包导入；没有删除代码、配置、数据、权重或实验产物。

| 原路径 | 新路径 | 原因 |
|---|---|---|
| `datasets/nuplan_dataset.py` | `datasets/source_contract.py` | 实现 source 版本与输入身份检查，并非 nuPlan Dataset |
| `datasets/core.py` | `datasets/preprocessing.py` | 集中几何预处理、轨迹采样及 split 工具，名称具体化 |
| `datasets/paired_dataset.py` | `datasets/pair/validation.py` | 仅包含 pair 校验，放入已有 pair 目录 |
| `datasets/representation_bridge.py` | `datasets/trajectory_resampling.py` | 数据侧轨迹重采样，与模型侧 bridge 区分 |
| `models/representation_bridge.py` | `models/source_trajectory_bridge.py` | 模型侧 source 轨迹桥接，与数据侧重采样区分 |
| `losses/research_score.py` | `losses/objectives.py` | 集中现有训练目标，取消流程名称作为文件名 |
| `datasets/audit.py` | `datasets/archive/audit.py` | 纯 re-export；当前调用直接引用 preprocessing，原文件归档 |
| `datasets/split_builder.py` | `datasets/archive/split_builder.py` | 纯 re-export；当前调用直接引用 preprocessing，原文件归档 |
| `datasets/trajectory_spec.py` | `datasets/archive/trajectory_spec.py` | 纯 re-export；当前调用直接引用 preprocessing，原文件归档 |
| `scripts/validate_native_v122.py` | `scripts/validate_native_transfer_manifests.py` | 实际检查 v1.2.3 manifest，名称与构建入口对应 |
| `scripts/build_native_v122.py` | `scripts/build_native_transfer_manifests.py` | 实际 REV 为 v1.2.3，去除误导性的 v122 名称 |
| `scripts/train_method_formal.py` | `scripts/train_transfer_early_stopping.py` | 按 validation loss 与 patience 选择 checkpoint |
| `scripts/train_method_equal_epochs_fine_select.py` | `scripts/train_transfer_fixed_updates.py` | 固定更新预算，保留独立选择与 scheduler 规则 |
| `scripts/build_manifests.py` | `scripts/archive/build_manifests.py` | 旧 Pluto cache manifest 流程 |
| `scripts/build_remediation.py` | `scripts/archive/build_remediation.py` | 旧 Pluto remediation 流程 |
| `scripts/diagnose_length_recovery.py` | `scripts/archive/diagnose_length_recovery.py` | 旧 remediation 长度诊断 |
| `scripts/validate_remediation.py` | `scripts/archive/validate_remediation.py` | 旧 remediation 验证 |
| `scripts/audit_routes_remediation.py` | `scripts/archive/audit_routes_remediation.py` | 旧 remediation 路线审计 |
| `scripts/build_nonoverlap8m_test.py` | `scripts/archive/build_nonoverlap8m_test.py` | 与 build_nonoverlap_segments.py 内容相同；当前入口统一到后者 |
| `configs/experiment/base.yaml` | `configs/experiment/archive/base.yaml` | 与 stage02.yaml 内容相同；当前配置使用 stage02.yaml |
| `configs/experiment/stage03_v1_2.yaml` | `configs/experiment/archive/stage03_v1_2.yaml` | 旧 Pluto v1.2 配置 |
| `configs/experiment/stage03_v1_2_1.yaml` | `configs/experiment/archive/stage03_v1_2_1.yaml` | 旧 Pluto v1.2.1 配置 |
| `configs/experiment/stage03_v1_2_2.yaml` | `configs/experiment/archive/stage03_v1_2_2.yaml` | 旧版本配置，不作为当前 native v1.2.3 默认配置 |
| `experiments/run_stage03.sh` | `experiments/archive/run_stage03.sh` | 旧 Pluto Stage 03 启动脚本 |
| `experiments/run_stage07_seed11.sh` | `experiments/archive/run_stage07_seed11.sh` | 早期 300-step smoke 实验启动记录 |
| `experiments/run_stage07_seed11_retry6.sh` | `experiments/archive/run_stage07_seed11_retry6.sh` | 早期 retry 实验启动记录 |

## 合并与引用调整

- 三个 re-export 文件没有独立算法；当前调用直接从 `datasets.preprocessing` 导入原定义，原 facade 放入 `datasets/archive/` 保留。
- 两个 non-overlap 构建脚本原内容完全相同；当前调用统一使用 `python -m scripts.build_nonoverlap_segments`，另一份归档。未合并不同评估协议。
- `base.yaml` 与 `stage02.yaml` 原内容完全相同；保留 `stage02.yaml` 为当前引用，另一份归档，配置内容未修改。
- 更新 Python import、测试文件中的引用与 mock 目标、shell 命令、配置路径以及当前文档引用。归档 `run_stage03.sh` 的仓库根定位随目录深度调整。
- 核心文档继续按八个主题维护；只更新实际路径、统一部分标题、将 preflight 示例改为现有 Stage 02 配置。六份历史文档原文不改写，分类仍由 [文档索引](../README.md) 统一维护。

迁移后的命令使用新模块名，例如 `python -m scripts.train_transfer_early_stopping`、`python -m scripts.train_transfer_fixed_updates`。原命令参数保持不变，运行说明见 [训练流程](../05_training.md)。归档脚本可通过 `python -m scripts.archive.<模块名>` 寻址，但仍需要对应历史输入；归档不表示其适用于当前 native 数据。

## 暂时保留与未解决事项

| 文件或模块 | 保留原因 / 尚未解决问题 |
|---|---|
| `datasets/manifests.py`、`datasets/tartan_dataset.py`、`datasets/trajectory_resampling.py` | 仓库内直接调用较少或未确认，但属于可独立使用的数据接口；不能排除外部调用 |
| `configs/experiment/proxy_smoke_v1_3_1.yaml`、`configs/experiment/stage06_methods.yaml` | 协议声明；未确认自动读取入口不等于无效 |
| `experiments/` 下不同 recipe | 预算、早停、选择及评估协议不同，不能按相似文件名合并；服务器绝对路径保留 |
| `datasets/preprocessing.py` | 多类相关工具，但文件规模有限；本轮不为了分类进一步拆分 |
| `models/diffusion/decoder.py` 与 `models/components/dit.py` | 前者组装 Decoder/RouteEncoder/DiT，后者提供基础组件；保留模块属性以避免 checkpoint 参数路径变化 |
| `diffusion_planner/`、`pair/`、`tartan/` 残留目录及 `tartan/.gitignore` | 主要为空目录或忽略的 bytecode；未清除、未判定未知本地产物可删除 |
| `nuplan_train.json`、`normalization.json`、`checkpoints/args.json` | 现有数据血缘或 checkpoint 配置；保留原路径与内容 |

没有抽象通用 Trainer 或合并不同训练循环；不同方法的 Loss、选择规则和 scheduler 时钟继续独立。当前 native 配置与通用 preflight 的 `paths` 契约仍不同，本轮仅修正文档示例，不扩展配置或修改检查逻辑。

## 仅记录的潜在算法边界问题

以下来自代码审查，尚未用真实输入确认；本轮未修改：

- `datasets/preprocessing.py::assert_no_split_leak`：查询使用原值、记录使用字符串；非字符串 ID 可能影响重复检测。
- `evaluation/goal_benchmark.py::evaluate_policy`：未知 policy 分支可能访问未赋值的路径变量。
- `datasets/route_builder.py::OccupancyRouteSetBuilder.build`：空稀疏输入可能在最大值计算处失败。

## 验证与兼容性

按用户此前要求，只做静态检查，不运行测试、训练、评估或服务器命令。检查 Python AST、本地 import 与 from 符号、shell `python -m` 目标、当前文档相对链接，并与本轮前快照核对代码与配置差异。独立快照复核结果：186 个基线文件均有对应文件；141 个原 Python 文件排除批准的 import 与说明文本变动后，非 import AST 一致；10 个 YAML、3 个受保护 JSON 和六份旧归档文档 hash 一致；8 个训练/评估入口的 argparse 参数与默认值一致。当前 145 个 Python 文件通过语法检查，本地导入目标与符号、18 个 shell/script 模块调用目标和当前文档相对链接可解析。静态检查不能保证服务器依赖、真实数据或 checkpoint 加载成功。

类名、函数名与模型属性保持不变；不保留旧 import/CLI 兼容入口，外部启动命令需要改用新路径。已有数据路径、输出目录、实验 ID、split、seed 与超参数不调整。代码中的 import 或说明文本修改会改变按源码内容计算的 provenance hash；单纯文件改名不会改变内容 hash。已有 manifest、缓存和 checkpoint 未重写或重新生成。

## 后续精简：删除空目录与重复文件

本节记录用户随后明确授权的删除与合并；上面的目录树、26 项迁移表和验证数量是第二轮整理完成时的历史快照。

- 删除 `scripts/archive/build_nonoverlap8m_test.py`：与当前 `scripts/build_nonoverlap_segments.py` 字节完全相同，当前调用已使用后者。上表中的该归档目标已在本次精简删除。
- 删除 26 个空目录，包括 `diffusion_planner/` 的空目录树、`tartan/research_score/` 的空目录树及其他空子目录。使用 `rmdir`，不删除目录内文件，不处理 `.git/`；含 `.gitignore` 或 bytecode 的旧目录仍保留。
- 合并以下五个小测试文件并删除原文件，全部测试函数与断言保留，没有删减独立测试覆盖。

| 删除的原测试文件 | 测试内容合入 |
|---|---|
| `tests/test_shortest_path.py` | `tests/test_route_replanning.py` |
| `tests/test_length_selection.py` | `tests/test_splits.py` |
| `tests/test_route_no_future.py` | `tests/test_route_set.py` |
| `tests/test_goal_no_future.py` | `tests/test_route_replanning.py` |
| `tests/test_canonicalize.py` | `tests/test_stage03_single_point.py` |

`scripts/` 的 Python 文件从 44 减至 43，`tests/` 从 26 减至 21（均含 `__init__.py`）。其他训练、评估、旧数据协议和诊断入口存在不同的输入、输出或选择规则，不能仅按相近名称删除；本轮保留。直接按旧测试文件路径启动的外部命令需要改用对应合入路径。验证仅采用静态语法、引用和迁移前后测试函数 AST 对比，没有运行测试或服务器任务。


## 后续收敛：只保留导航选择训练与 segment 模型评估

用户进一步明确保留 `scripts/train_finetune_navigation_earlystop.py` 和 `scripts/evaluate_nonoverlap_segments.py`。前文的多训练、多评估入口表述属于修改前历史状态。

- 删除四个旧训练入口：`train_method_smoke.py`、`train_target_only.py`、`train_transfer_early_stopping.py`、`train_transfer_fixed_updates.py`（均原位于 `scripts/`）。
- 删除七个专属训练 launcher：`experiments/run_stage07_formal_seed11.sh`、`run_stage07_formal_seed11_v3.sh`、`run_stage07_d029_seed11.sh`、`run_finetune_20_50_seed11.sh`、`run_finetune_equal100epochs_seed11.sh`，以及 `experiments/archive/run_stage07_seed11.sh`、`run_stage07_seed11_retry6.sh`。
- 删除 `scripts/scan_snapshot_validation.py`：用于旧训练产物的 validation-Loss snapshot 排名，不再提供该选择流程。
- 删除 `scripts/evaluate_model_closed_loop.py`、`scripts/evaluate_model_closed_loop_nonoverlap.py`，以及 classic 独占的 `experiments/run_stage07_formal_eval_seed11.sh`、`scripts/summarize_formal_seed11.py`。
- `experiments/run_stage07_v124_nonoverlap_eval.sh` 改为调用保留的 `scripts.evaluate_nonoverlap_segments`，其余参数、数据路径和输出路径不变。
- 四个共享训练函数 `load_ckpt`、`loss_step`、`validate`、`publish` 与原 `ABILITIES` 常量迁到 `engine/training_utils.py`，导航训练只更改 import；函数体、模型结构、损失、训练默认值和 checkpoint 内容保持不变。

保留的训练仍用 validation Loss 调节学习率；最佳 checkpoint 和早停按导航表现判断。保留的模型 evaluator 对清单中的每个 segment 独立 rollout，再按 episode 汇总。其源码未修改；它优先采用 checkpoint 的 `ema_state_dict`，其次 `model`，因此含两套不同权重的旧 checkpoint 与原 model-only evaluator 可能产生不同预测。导航训练产出的 `navigation_best.pt` 保存的是 `model` 字段。

共享评估模块和 `scripts/run_stage03b.py` 参考基线审计仍保留；它不是另一种模型 checkpoint evaluator。历史配置、可独立读取已有结果的分析脚本及使用保留 evaluator 的旧结果评估 recipe 仍保留。数据、权重、manifest 与实验结果均未删除或重新生成。当前运行说明见 [训练流程](../05_training.md)与[评估与推理](../06_evaluation.md)。


## 最终入口收敛

用户再次确认只保留导航训练与 segment 模型评估。删除以下旧版本配套文件，前文保留历史评估 recipe/汇总工具的描述已不代表当前树：

- `experiments/run_finetune_equal100epochs_eval.sh`
- `experiments/run_finetune_equal100epochs_final_eval.sh`
- `experiments/run_stage07_d029_eval.sh`
- `experiments/run_stage07_nonoverlap_eval.sh`
- `experiments/run_stage07_v124_nonoverlap_eval.sh`
- `scripts/summarize_nonoverlap_segments.py`
- `scripts/summarize_v124_nonoverlap.py`
- `scripts/summarize_stage07_first_round.py`
- `scripts/summarize_experiment1_equal_epochs.py`
- `configs/experiment/stage06_methods.yaml`
- `configs/experiment/proxy_smoke_v1_3_1.yaml`

当前训练入口为 `scripts/train_finetune_navigation_earlystop.py`，模型评估入口为 `scripts/evaluate_nonoverlap_segments.py`，唯一模型评估 shell 为 `experiments/run_finetune_navigation_earlystop_eval.sh`。训练直接使用 CLI，没有新增 shell 或兼容入口。独立参考基线审计、数据工具和共享模块保留；已有数据、checkpoint 与结果仍未删除。上述保留入口与共享训练函数的源码内容未修改。


## 主入口目录精简

用户要求 `scripts/` 仅保留主要运行入口。本次将数据、审计和诊断工具按职责迁出，并将两个主入口改名。前文入口路径属于历史状态；当前命令为 `python -m scripts.train` 与 `python -m scripts.evaluate`，参数不变。没有独立 inference 流程，不创建占位 `inference.py`；模型推理仍由评估 rollout 执行。

```text
scripts/
├── __init__.py
├── train.py
└── evaluate.py

datasets/
├── tools/       # 数据构建、缓存物化、数据审计
├── pair/        # pair 工具及 mining 入口
└── archive/     # 历史数据流程

evaluation/run_stage03b.py  # 独立参考基线审计
experiments/analysis/      # source、proxy、fixture 与 CPU 诊断
```

| 原路径 | 新路径 |
|---|---|
| `scripts/train_finetune_navigation_earlystop.py` | `scripts/train.py` |
| `scripts/evaluate_nonoverlap_segments.py` | `scripts/evaluate.py` |
| `scripts/run_stage03b.py` | `evaluation/run_stage03b.py` |
| `scripts/build_d029_budgets.py` | `datasets/tools/build_d029_budgets.py` |
| `scripts/build_d029_extended_budgets.py` | `datasets/tools/build_d029_extended_budgets.py` |
| `scripts/build_native_transfer_manifests.py` | `datasets/tools/build_native_transfer_manifests.py` |
| `scripts/validate_native_transfer_manifests.py` | `datasets/tools/validate_native_transfer_manifests.py` |
| `scripts/audit_native_loader_identity.py` | `datasets/tools/audit_native_loader_identity.py` |
| `scripts/materialize_source_features.py` | `datasets/tools/materialize_source_features.py` |
| `scripts/materialize_target_features.py` | `datasets/tools/materialize_target_features.py` |
| `scripts/build_nonoverlap_segments.py` | `datasets/tools/build_nonoverlap_segments.py` |
| `scripts/audit_routes.py` | `datasets/tools/audit_routes.py` |
| `scripts/audit_pairs.py` | `datasets/tools/audit_pairs.py` |
| `scripts/audit_source_model.py` | `experiments/analysis/audit_source_model.py` |
| `scripts/pair/run_pair_mining.py` | `datasets/pair/run_pair_mining.py` |
| `scripts/archive/build_manifests.py` | `datasets/archive/build_manifests.py` |
| `scripts/archive/build_remediation.py` | `datasets/archive/build_remediation.py` |
| `scripts/archive/diagnose_length_recovery.py` | `datasets/archive/diagnose_length_recovery.py` |
| `scripts/archive/validate_remediation.py` | `datasets/archive/validate_remediation.py` |
| `scripts/archive/audit_routes_remediation.py` | `datasets/archive/audit_routes_remediation.py` |
| `scripts/build_proxy_cpu_inputs.py` | `experiments/analysis/build_proxy_cpu_inputs.py` |
| `scripts/materialize_fixture_pairs.py` | `experiments/analysis/materialize_fixture_pairs.py` |
| `scripts/validate_proxy_cpu_inputs.py` | `experiments/analysis/validate_proxy_cpu_inputs.py` |
| `scripts/run_proxy_smoke.py` | `experiments/analysis/run_proxy_smoke.py` |
| `scripts/run_proxy_swap.py` | `experiments/analysis/run_proxy_swap.py` |
| `scripts/run_proxy_cpu_forward.py` | `experiments/analysis/run_proxy_cpu_forward.py` |
| `scripts/run_fixture_cpu_loss.py` | `experiments/analysis/run_fixture_cpu_loss.py` |
| `scripts/visualize_fixture_pairs.py` | `experiments/analysis/visualize_fixture_pairs.py` |

原 `scripts/pair/` 和 `scripts/archive/` 的空重复 package marker 合并到已存在的数据包标记文件，删除空旧目录；当前 `scripts/__init__.py` 保留用于模块启动。数据与工具实现保留，仅路径、import、启动命令和当前文档引用更新；数据路径、参数、checkpoint 与产物内容不变。历史文档原文不重写。
