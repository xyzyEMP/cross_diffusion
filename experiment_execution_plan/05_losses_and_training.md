---
stage: 05
plan_version: "1.2.3"
status: COMPLETE
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_04]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 05：损失函数与分阶段训练管线

> v1.2.2 边界：文中遗留的 Proxy 执行命令仅可改写为 synthetic fixture 测试，不得启动真实数据训练。正式 `L_inv/L_swap` 实验等待 Strict 数据。

> 前置：[阶段 04](./04_model_decomposition.md) 模型接口已确认。  
> 下一阶段只做基线公平性 smoke，不直接跑正式矩阵。

## 1. 阶段目标

实现并验证 `L_diff + λ_inv L_inv + λ_swap L_swap + λ_sep L_sep`，保证每项 loss 的输入、目标平台、噪声和梯度路径正确，再建立可恢复、可复现的训练脚本。

训练入口必须区分：

- `transfer_primary`：nuPlan Car checkpoint→ANYmal，只运行不要求严格 pair 的迁移方法；不得把 Diff↔ANYmal 的 `L_inv/L_swap` 加入主迁移 run；
- `proxy_pair_auxiliary`：仅运行 synthetic/interface fixture 的公式与接口测试，不训练模型、不形成正式 loss 结果；
- `strict_pair`：未来 Car↔Dog 严格 pair，运行完整 loss 并支持正式方法结论。

## 2. 四类损失

### 2.1 基础扩散损失 `L_diff`

保留 checkpoint 的 `x_start` 参数化：

$$
\mathcal L_{diff}^e
=\mathbb E\left[m\left\|\hat x_{0,I}^e+\Delta\hat x_{0,E}^e-x_0^e\right\|_2^2\right].
$$

配对 profile 的平台 A/B batch 均衡采样，不能因某侧数据更多而改变隐式权重。`transfer_primary` 的 nuPlan/ANYmal 混合比例是方法配置的一部分，所有相应基线使用同一规则。邻居 loss 和 ego loss 权重保持原配置；无邻居标注时正确 mask。源 observation/state normalizer 冻结。

### 2.2 共享一致性 `L_inv`

Pair 使用同一个 canonical noisy query、同一 $t$ 和同一噪声。第一版在 `x0` 空间比较共享预测：

$$
\mathcal L_{inv}=\mathbb E\left[m_{pair}\left\|\hat x_{0,I}(q_t,c_I^{A})-\hat x_{0,I}(q_t,c_I^{B})\right\|_2^2\right].
$$

正式训练只接受 `strict_pair` 中 `pair_valid=true` 的真实配对。`proxy_pair_auxiliary` 仅用 synthetic fixture 验证张量方向、梯度和交换接口，不输出模型效果；`strict_pair` 才支撑 Car–Dog 训练与结论。两条轨迹轮流作为 query 来源，避免固定偏向 A 或 B。epsilon/score 距离只作为派生诊断，并必须带显式 `sigma_clamp` 与 timestep 分桶。

### 2.3 定向交换 `L_swap`

A→B 必须以 B 的目标轨迹加噪，并用 B target 监督：

$$
\hat x_0^{A\rightarrow B}
=\hat x_{0,I}(q_t^{B},c_I^{A})
+\Delta\hat x_{0,E}(q_t^{B},c_E^{B}),
$$

完整损失为：

$$
\mathcal L_{swap}^{A\rightarrow B}
=\mathbb E\left[m_B\left\|\hat x_0^{A\rightarrow B}-x_0^B\right\|_2^2\right],
\qquad
\mathcal L_{swap}=\mathcal L_{swap}^{A\rightarrow B}+\mathcal L_{swap}^{B\rightarrow A}.
$$

反向交换以 A 轨迹为 query/target。禁止对一个未说明来源的公共 epsilon 同时监督两个方向。

### 2.4 分离约束 `L_sep`

第一版落实为两项：

- `L_adv`：仅在场景匹配 pair 的共享表示上接具身分类器与 Gradient Reversal；
- `L_res`：约束残差幅值，并通过小容量 adapter 限制具身分支吞掉全部预测。

$$
\mathcal L_{sep}=\lambda_{adv}\mathcal L_{adv}+\lambda_{res}\mathcal L_{res}.
$$

不强制共享与具身向量正交，除非后续实验证明有必要。

所有样本显式记录 `dataset_id/domain_id/embodiment_id`。训练外另做 domain probe；如果共享表示仍主要识别数据域，不能把 GRL 结果解释成具身解耦。

## 3. 代码落点

```text
tartan/research_score/training/
├── losses.py
├── paired_batch.py
├── samplers.py
├── trainer.py
├── schedules.py
├── checkpointing.py
└── train.py
tartan/research_score/scripts/
├── train_stage1_adapter.sh
├── train_stage2_paired.sh
├── train_stage3_joint.sh
└── run_stage05_tiny.sh
tartan/research_score/tests/
├── test_losses.py
├── test_paired_noise.py
├── test_gradient_routing.py
├── test_resume.py
└── test_tiny_overfit.py
```

## 4. 训练阶段

1. **Adapter 预适配**：冻结共享主干，用平台均衡普通样本训练零初始化具身 adapter，仅启用 `L_diff`；
2. **配对解耦**：`proxy_pair_auxiliary` 只做 `L_inv/L_swap/L_sep` 的 synthetic 单元测试；`strict_pair` 数据通过审计后才运行正式训练；
3. **低学习率联合训练**：只解冻共享主干最后 1 个 DiT block 和 shared final layer；
4. **模型选择**：只用 validation 的综合指标选择 checkpoint；测试集不参与 early stopping。

初始超参可从 `lambda_inv=0.2`、`lambda_swap=0.5`、`lambda_sep=0.1` 开始，但必须视为搜索起点，最终搜索范围和选择准则写入配置。

## 5. validation 选择准则

不能只按 ADE 选模型。建议预注册：

```text
primary: validation goal success
tie-break 1: lower collision rate
tie-break 2: higher SPL
constraint: shared task sensitivity > minimum threshold
constraint: no NaN, no branch collapse
```

阈值只由 validation 与 tiny diagnostics 确定，并写入 `selection_rule.yaml`。

## 6. 必须通过的测试

- `L_diff` 在 total prediction 等于 GT 时为 0；
- pair 两侧确实共享同一 `q_t/t/noise`；
- A→B 的 target 是 B，B→A 的 target 是 A；
- invalid pair 对 `L_inv/L_swap` 梯度为 0；
- GRL 对 classifier 与 shared encoder 的梯度方向相反；
- GRL batch 只含匹配场景 pair，domain probe 单独报告；
- epsilon 派生诊断按 timestep 分桶，在小 $\sigma$ 下无 NaN/爆炸；主辅助 loss 不依赖该换算；
- 冻结阶段只有 adapter 参数更新；
- resume 后 optimizer、scheduler、EMA、step 和随机状态连续；
- 32–128 个样本 tiny set 可显著过拟合；
- 打乱目标或 pair 后 loss/指标按预期恶化，避免实现根本没使用条件。
- `transfer_primary` 尝试启用 Proxy pair loss 时必须被配置校验拒绝；
- 三 profile 产生不同 output namespace 和 claim scope；
- 重复 `run_id` 必须失败，同一 run 的显式 resume 必须恢复完整随机状态。

## 7. 实际执行

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m pytest tartan/research_score/tests/test_losses.py \
  tartan/research_score/tests/test_paired_noise.py \
  tartan/research_score/tests/test_gradient_routing.py \
  tartan/research_score/tests/test_resume.py \
  tartan/research_score/tests/test_tiny_overfit.py -q

# v1.2.2：上述测试中的配对项只允许 synthetic/interface fixture。
# 不得以 proxy_pair_auxiliary 启动真实数据训练。
```

## 8. 阶段产物与验收

- loss 单元测试报告；
- tiny-set loss 曲线；
- 每个 loss 的梯度范数；
- 分阶段可训练参数清单；
- resume 前后损失连续性；
- pair shuffle 负对照；
- domain/embodiment probe 诊断；
- `run_registry.parquet`，包含成功、失败、中断状态及潜在消融标签；
- `review_packet.md`；
- `stage_report.md`。

若 tiny set 无法过拟合、pair shuffle 不影响结果、residual 始终为零或 shared probe 已坍缩，本阶段不得通过。

验收前停止，不启动所有方法的正式训练。
