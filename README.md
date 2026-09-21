# Cross-Diffusion: nuPlan Car → TartanGround ANYmal
本仓库实现一个跨具身轨迹规划研究：将 nuPlan Car 扩散规划器迁移至 TartanGround ANYmal，输出机体级 SE(2) 路径。
目标域数据按完整 episode 切分为 train/validation/test = 16/3/5。训练使用8 m/80点密集滑窗；正式闭环测试使用5条 held-out episode 中构建的63个非重叠8 m 任务。
正式比较的四种方法：

- `pretrain_finetune`：加载 Car checkpoint，用 ANYmal 全参数微调。
- `pretrain_adapter`：冻结 Car 主干，仅训练具身条件适配器。
- `joint_train`：Car 和 ANYmal batch 交替更新同一主干。
- `emb_cond_diffusion`：在联合训练上加入平台 ID 、能力向量和每个去噪步骤的具身残差修正。

## 目录

```text
diffusion_planner/             原始 Diffusion-Planner 主干与扩散模块
tartan/
  research_score/
    configs/                   数据与协议配置
    data/                      表示、路线与数据契约
    model/                     具身条件编码和残差 adapter
    training/                  缓存数据集与训练辅助代码
    evaluation/                闭环、碰撞、路线和指标
    scripts/                   可复现的构建、训练和评价入口
    tests/                     协议和核心逻辑测试
docs/                          人类可读的模型、数据和结果总览
experiment_execution_plan/     冻结协议、风险与阶段状态
run.md                         当前正式复现命令
```

## 当前结果

seed 11 上，`pretrain_finetune` 在100% ANYmal 训练预算下获得最高的段级 SR 0.6349 和 SPL 0.5966。完整结果、解读和局限见 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)。

## 运行

依赖与路径参见 [run.md](run.md)。大型 checkpoint、特征缓存、数据和运行产物不纳入 Git，统一存放于共享工作盘。
