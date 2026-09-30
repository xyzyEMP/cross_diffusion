# Cross-Diffusion 训练与测试协议

本文档总结当前仓库中实际可执行的正式训练与测试设置。训练协议为
`score-decomp-transfer-v1.2.3`，正式训练入口为 v3；当前闭环测试任务为
v1.2.4。项目范围和完成状态分别见 [README.md](README.md) 与
[PROJECT_STATUS.md](PROJECT_STATUS.md)。

## 1. 实验目标与边界

实验研究将 nuPlan 预训练的 Diffusion Planner 迁移到 TartanGround ANYmal
局部导航。模型输出是机体级局部轨迹，不是关节或电机控制信号。

当前正式实验比较四种迁移方法，验证场景限定为同一地图上的未见 episode。
当前结果不是多随机种子结果，也不支持未见地图、真实机器人或因果解耦结论。

## 2. 数据与轨迹表示

### 2.1 数据域

- 源域：nuPlan / Diffusion Planner car 数据与预训练权重
  `checkpoints/model.pth`。
- 目标域：TartanGround ANYmal 轨迹和 occupancy map。
- 正式训练读取预生成缓存：
  `target_train_features.pt`、`target_val_features.pt` 和
  `source_car_features_4096.pt`。

源域缓存默认从 manifest 确定性抽取 4,096 个样本，默认抽样种子为
`20260914`。当前仓库没有记录正式缓存的文件哈希，也没有一条完整命令可证明
现有缓存由这些默认参数生成，因此复现实验前必须单独核验缓存来源。

### 2.2 目标轨迹

- 历史：20 帧，10 Hz。
- 未来表示：固定 8 m 弧长，重采样为 80 点。
- 单点表示：`[x, y, cos(yaw), sin(yaw)]`。
- 对不足 8 m 的有效部分使用 validity mask。
- 候选 anchor 默认每 10 帧提取一次。

源域原始目标是固定时间范围，目标域是固定空间弧长。两者经过表示桥接后进入
同一训练接口，但语义并不完全等价；涉及联合训练或共享机制的结论必须保留此限制。

相关实现：

- [`configs/stage03_v1_2_3.yaml`](../tartan/research_score/configs/stage03_v1_2_3.yaml)
- [`data/core.py`](../tartan/research_score/data/core.py)
- [`scripts/build_native_v122.py`](../tartan/research_score/scripts/build_native_v122.py)
- [`scripts/materialize_target_features.py`](../tartan/research_score/scripts/materialize_target_features.py)
- [`scripts/materialize_source_features.py`](../tartan/research_score/scripts/materialize_source_features.py)

## 3. 数据划分与预算

目标域按完整 episode 划分，目标比例为 70/15/15，划分种子为
`20260911`。当前文档记录的数据规模如下：

| 划分 | Episode 数 | 用途 |
|---|---:|---|
| Train | 16 | 2,048 个 dense windows |
| Validation | 3 | loss 监控与 35 个冻结导航任务 |
| Test | 5 | 63 个冻结、非重叠 8 m 任务 |

嵌套预算定义为：

| 预算 | 1% | 10% | 20% | 50% | 100% |
|---:|---:|---:|---:|---:|---:|
| 训练窗口 | 20 | 205 | 410 | 1,024 | 2,048 |

当前正式 v3 runner 只运行 1%、10% 和 100%；20% 与 50% 虽被训练器接受，
但不在该 runner 的任务矩阵中。所有预算应覆盖全部 16 个训练 episode，验证集和
测试集保持 episode 隔离。

## 4. 比较方法

| 方法 | 初始化 | 目标数据 | 源数据 | 可训练部分 |
|---|---|---:|---:|---|
| `pretrain_finetune` | nuPlan checkpoint | 是 | 否 | 完整 backbone |
| `pretrain_adapter` | nuPlan checkpoint | 是 | 否 | 冻结 backbone，仅训练 embodiment residual wrapper |
| `joint_train` | nuPlan checkpoint | 是 | 是 | 完整 backbone |
| `emb_cond_diffusion` | nuPlan checkpoint | 是 | 是 | backbone 与 embodiment residual wrapper |

wrapper 使用平台 ID 和 10 维能力向量，目标域 ID 为 1，源域 ID 为 0；residual
末层零初始化。

`additive_diff_only` 仅用于合成诊断；`ours_score_decomp` 需要严格跨形态配对，
当前标记为 `blocked_until_strict_pairs`，不属于正式训练矩阵。

相关实现：

- [`training/method_factory.py`](../tartan/research_score/training/method_factory.py)
- [`model/score_decomposition.py`](../tartan/research_score/model/score_decomposition.py)
- [`configs/stage06_methods.yaml`](../tartan/research_score/configs/stage06_methods.yaml)

## 5. 正式训练设置

### 5.1 损失与更新

正式训练只使用归一化未来轨迹上的 masked denoising MSE：

```text
L = sum(mask * ||prediction - normalized_target||^2) / sum(mask)
```

正式训练不使用 `L_inv`、`L_swap` 或 `L_sep`。虽然输出字段名为 `score`，当前
训练目标是归一化轨迹，因此本文不把它解释为已经识别的因果 score。

对 `joint_train` 和 `emb_cond_diffusion`，每次 target optimizer update 后，再从
源域缓存有放回抽取一个同 batch 大小的数据批次，执行一次独立 source update。

### 5.2 超参数

| 项目 | 设置 |
|---|---|
| Batch size | 64 |
| 最大 target updates | 10,000 |
| 最小 target updates | 5,000 |
| 验证间隔 | 每 250 个 target updates |
| Early-stopping patience | 连续 10 次验证无改善 |
| 改善阈值 | `1e-5` |
| Optimizer | AdamW |
| Adapter-only learning rate | `2e-4` |
| 其他方法 learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Gradient clipping | 5.0 |
| AMP | 启用 |
| 正式训练 seed | 11 |
| 固定验证 seed | `20260914` |

学习率调度器为 `ReduceLROnPlateau`，factor 为 0.5，patience 为 2，最小学习率为
`5e-6`。

### 5.3 验证与 checkpoint

`best.pt` 按最低 validation denoising loss 选择，不按闭环导航指标选择。
训练同时输出：

- `best.pt`：最低验证损失对应权重；
- `last.pt`：最终模型及优化器、scaler、scheduler 状态；
- `metrics.json`：训练设置、更新次数、最佳验证损失和训练历史。

实现见
[`scripts/train_method_formal.py`](../tartan/research_score/scripts/train_method_formal.py)。

## 6. 正式测试协议

当前测试入口为
[`run_stage07_v124_nonoverlap_eval.sh`](../tartan/research_score/scripts/run_stage07_v124_nonoverlap_eval.sh)。
它加载正式训练目录中的 `best.pt`，对四种方法和 1%、10%、100% 三档预算进行
评估，共 12 个组合。

### 6.1 测试任务

- 测试任务来自完整隔离的 test episode。
- 每个任务沿记录轨迹向前取 8 m 终点作为固定目标。
- 相邻任务从不早于上一段终点的 anchor 开始，以避免局部任务重叠。
- 协议文档记录 63 个任务，但生成后的正式 manifest 未提交到仓库，因此实际运行
  时必须以 manifest 条目数为准。
- 路线特征构造不直接读取未来轨迹，但目标点来自记录轨迹的未来 8 m，因此仍包含
  goal-selection oracle，不应称为完全无 oracle 的自主导航测试。

### 6.2 闭环 rollout

| 项目 | 设置 |
|---|---|
| 起始状态 | 局部原点、yaw 为 0 |
| 候选路线 | 每次最多 6 条 |
| 单次执行轨迹 | 最多 1 m |
| 插值间隔 | 0.1 m |
| 最大 replans | 16 |
| 成功阈值 | 到目标距离不超过 0.75 m |
| Stuck 判定 | 连续 3 次移动不足 0.05 m |
| Collision | 发生即终止 |

每次 rollout 使用确定性推理种子
`10000 + episode_index * 100 + replan_index`，没有对多次 diffusion draw 取平均。

### 6.3 指标与汇总

每个 segment 记录：

- success rate（SR）；
- collision rate（CR）；
- SPL；
- travelled path 与 shortest path；
- goal progress；
- stuck rate；
- route-failure rate；
- replans、终止原因和平均推理时间。

SPL 定义为：成功时
`shortest_path / max(shortest_path, travelled_path)`，失败时为 0。Goal progress
定义为 `(初始目标距离 - 最终目标距离) / 初始目标距离`。

汇总同时报告 segment 微平均和 episode-macro 平均。无效地图 episode 不进入指标
分母，其余失败（包括 route failure）保留在分母中。

相关实现：

- [`scripts/build_nonoverlap8m_test.py`](../tartan/research_score/scripts/build_nonoverlap8m_test.py)
- [`scripts/evaluate_model_closed_loop_nonoverlap.py`](../tartan/research_score/scripts/evaluate_model_closed_loop_nonoverlap.py)
- [`evaluation/metrics_navigation.py`](../tartan/research_score/evaluation/metrics_navigation.py)
- [`scripts/summarize_v124_nonoverlap.py`](../tartan/research_score/scripts/summarize_v124_nonoverlap.py)

## 7. 执行入口

```bash
bash tartan/research_score/scripts/run_stage07_formal_seed11_v3.sh
bash tartan/research_score/scripts/run_stage07_v124_nonoverlap_eval.sh
```

这些脚本写死了服务器路径、conda 环境和 GPU 编号，例如
`/zeron-vepfs/tjqc/cross-diffusion`、
`/tj-share/cross_diffusion_workdir/research_score_v1_2` 和
`/root/miniconda3/envs/diffusion-planner`。在其他机器执行前必须先按实际环境修改。

## 8. 当前已知不一致与复现缺口

1. 正式 runner 枚举 4 种方法 × 3 档预算，共 12 个 run，但结尾检查 15 个
   `metrics.json`；按当前循环完成全部任务后该检查仍可能失败。
2. `stage06_methods.yaml` 使用 `repeat_id: 44`，正式 runner 使用 seed 11；预算
   membership 的正式来源需要进一步确认。
3. v1.2.3 配置列出 11、22、33、44、55 五个计划 seed，正式 runner 当前只执行
   seed 11，不能称为多 seed 实验。
4. `tartan/research_score/README.md` 仍指向旧的
   `run_stage07_nonoverlap_eval.sh`；当前 v1.2.4 测试入口是
   `run_stage07_v124_nonoverlap_eval.sh`。
5. 现有 README/状态文档提到 navigation-metric checkpoint selection，但正式训练器
   实际按 validation denoising loss 保存 `best.pt`。
6. 仓库未提交正式 cache 的哈希和生成记录，也未提交可直接核验“63 个测试任务”的
   最终 manifest。
7. 配置声明 branch-stratified split，但当前执行入口消费已生成的数据，不能仅凭仓库
   代码确认服务器上的 manifest 确实由该算法生成。

## 9. 结果解释范围

当前协议可以支持：

- 同地图、未见 episode 上的跨域迁移比较；
- 四种正式迁移方法在固定数据预算下的性能比较；
- segment 和 episode-macro 粒度的闭环导航指标。

当前协议不能支持：

- 多 seed 稳定性；
- 未见地图泛化；
- 严格跨形态因果解耦或反事实预测；
- 真实机器人或物理可执行性；
- 完全无 oracle 的自主导航结论。
