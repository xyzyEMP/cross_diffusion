---
stage: 04
plan_version: "1.2.3"
status: COMPLETE
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_03B]
profiles: [proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 04：共享—具身 Score 分解模型实现

> v1.2.2 边界：Proxy 只用于 synthetic/interface 单元测试；不得加载真实 Diff↔ANYmal 作为配对训练数据。正式配对训练等待 Strict 数据。

> 前置：[阶段 03B](./03b_goal_navigation_benchmark.md) 的无泄漏目标导航协议已确认。  
> 本阶段只实现模型与推理接口，不启动正式训练。

## 1. 实现选择

飞书方案使用 epsilon/score 表达，但现有预训练 checkpoint 是 `x_start` 模型。为保留源模型能力，采用以下等价实现：

$$
\hat x_0^e=\hat x_{0,I}+\Delta\hat x_{0,E}.
$$

令 $a_t=\sqrt{\bar\alpha_t}$、$\sigma_t=\sqrt{1-\bar\alpha_t}$，则：

$$
\hat\epsilon_I=\frac{x_t-a_t\hat x_{0,I}}{\sigma_t},
\qquad
\hat\epsilon_E=-\frac{a_t\Delta\hat x_{0,E}}{\sigma_t},
$$

$$
\hat\epsilon_{total}=\hat\epsilon_I+\hat\epsilon_E
=\frac{x_t-a_t(\hat x_{0,I}+\Delta\hat x_{0,E})}{\sigma_t}.
$$

这样训练使用原生 `x_start` loss，并可导出 epsilon/score 作为诊断。由于换算含 $a_t/\sigma_t$，阶段 05 的辅助一致性与 Swap 第一版也统一在 `x0` 空间计算；禁止直接在未加权 epsilon 空间优化，也禁止把旧权重当作 score 模型。

## 2. 双空间边界与表示桥接

原 checkpoint 仍在 `source_temporal_8s_80` 空间做严格数值回归；研究训练与跨平台一致性在 `fixed_arc_length_L_80` 空间进行。新增 `TrajectoryRepresentationBridge` 明确执行弧长截取、80 点重采样、yaw 连续化、valid mask 和时间戳旁路。它不是无参数等价变换时，必须把可训练参数、训练数据和误差单独报告。

- `source_regression_mode`：完全绕过 bridge 和研究模块，复现阶段 02 签名；
- `research_mode`：源 backbone 特征经 bridge/adapter 输出固定弧长轨迹；
- 禁止在 `research_mode` 声称零残差等于原 8 s 输出；
- bridge 的训练只使用 train split；速度与执行时间由平台执行适配器恢复。

## 3. 模型结构

```text
统一场景/目标/历史 ── 原 Encoder + 原 DiT blocks ── x0_shared
                                   │
平台 ID + 能力向量 ─ embodiment encoder
                                   │
                  zero-init residual adapters ── delta_x0_embodiment
                                   │
                   x0_total = x0_shared + delta_x0_embodiment
```

第一版只对 ego trajectory 增加具身 correction；neighbor prediction 保持源模型共享输出，避免在 Tartan 无动态 agent 条件下引入无监督分支。

### 共享分支输入

- 统一场景、目标、历史状态和已冻结的 route 条件；
- 不输入 platform ID、平台尺寸、最大坡度、最大台阶等具身字段。

### 具身分支输入

- platform ID；
- footprint 长宽；
- 最大速度、角速度、加速度和曲率；
- 最大坡度、台阶、粗糙度/支撑阈值；
- 第一版无在线反馈，缺失字段由显式 mask 表示。

能力向量先做固定单位归一化，归一化统计只由训练集生成。

## 4. 代码落点

```text
tartan/research_score/model/
├── embodiment.py
├── representation_bridge.py
├── adapters.py
├── score_decomposition.py
├── outputs.py
└── checkpoint.py
tartan/research_score/tests/
├── test_parameterization.py
├── test_representation_bridge.py
├── test_zero_residual.py
├── test_model_shapes.py
├── test_condition_isolation.py
└── test_checkpoint_loading.py
```

推荐实现 `ScoreDecompositionPlanner`，返回具名结构：

```python
{
    "x0_shared": ...,
    "x0_embodiment_delta": ...,
    "x0_total": ...,
    "eps_shared": ...,
    "eps_embodiment": ...,
    "eps_total": ...,
    "z_shared": ...,
    "z_embodiment": ...,
}
```

采样器默认使用 `x0_total`。增加 `shared_context_override` 和 `embodiment_context_override`，供阶段 09 做定向 swap，但默认关闭。

## 5. checkpoint 兼容

- 原 Encoder、route encoder、DiT 和 shared final layer 从 `checkpoints/model.pth` 严格映射；
- 新增 embodiment encoder、adapter 和 residual head 零初始化；
- 提供映射报告，列出每个源权重去向；
- `source_regression_mode` 下输出必须与阶段 02 保存的 signature 一致；
- `research_mode` 单独报告 bridge 重建误差和固定弧长输出，不混作源模型复现；
- 保存 checkpoint 时分开记录 `source_state`、`research_state` 和优化器状态。

## 6. 必须通过的测试

1. **参数化恒等式**：`eps_total == eps_shared + eps_embodiment`，误差 `<1e-5`；
2. **源空间回归**：`source_regression_mode` 固定输入/seed 下与原模型误差 `<1e-6`；
3. **桥接正确性**：解析直线/圆弧重采样误差在配置容差内，mask、yaw 和时间戳旁路正确；
4. **输入隔离**：只改变 platform ID 时，`x0_shared/z_shared` 不变；
5. **具身响应**：手工设置非零 adapter 后，只改变能力向量会改变 residual；
6. **形状与 mask**：ego、neighbors、route-set、padding 和 batch size 1/4 均正确；
7. **数值稳定**：派生 epsilon 时对 $\sigma_t$ 使用显式配置的 clamp，仅用于诊断；比较 clamp 前后误差并确认不进入主辅助 loss；
8. **梯度检查**：共享 loss、bridge loss 与 residual loss 能到达各自参数；冻结开关有效；
9. **推理 smoke**：同一 checkpoint 分别生成源 8 s/80 帧回归输出和固定弧长/80 点研究输出，目录与指标明确分开。

## 7. 运行命令

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 04 --profile transfer_primary \
  --config tartan/research_score/configs/stage04.yaml \
  --report "$OUTPUT_ROOT/preflight/stage04"

$PYTHON_BIN -m pytest tartan/research_score/tests/test_parameterization.py \
  tartan/research_score/tests/test_representation_bridge.py \
  tartan/research_score/tests/test_zero_residual.py \
  tartan/research_score/tests/test_model_shapes.py \
  tartan/research_score/tests/test_condition_isolation.py \
  tartan/research_score/tests/test_checkpoint_loading.py -q

$PYTHON_BIN -m tartan.research_score.scripts.validate_model \
  --config tartan/research_score/configs/model.yaml \
  --output "$OUTPUT_ROOT/02_model_validation/proxy_pair_auxiliary"
```

## 8. 阶段产物与验收

产物：权重映射报告、参数量分解、公式误差 JSON、零残差回归 NPZ、GPU 内存/时延、固定样本输出、`review_packet.md` 和 `stage_report.md`。

人工只在以下条件全部满足时通过：源空间严格回归、桥接误差合格且未伪装为等价、shared 不读取具身字段、恒等式成立、双向 override 接口有单元测试、原 baseline 未被破坏。

验收前停止，不实现 loss 或训练循环。
