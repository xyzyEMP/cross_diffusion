# 数据与数据流程

模型运行使用已准备好的 target feature cache 与 navigation manifests。训练、验证和测试不在两个模型入口中重新生成 split 或缓存。通用路径和 CLI 参数见[实验与资源](07_experiments.md)；1_全量微调复现的固定输入核对和样本数量见[实验记录](experiments/1_全量微调复现.md)。

## 模型运行输入

`datasets/cached_target_dataset.py` 读取预生成的 cache；`datasets/preprocessing.py`、`datasets/route_builder.py` 和 `datasets/tartanground/features.py` 提供数据表示与路线/特征构造，normalizer 位于 `datasets/transforms/normalizer.py`。训练入口使用训练 cache 和 validation cache；导航验证/测试 manifest 引用任务与地图。缓存和 manifest 是既有固定输入，运行训练前不得重新生成替换。

source args 提供 Diffusion Planner 构造参数及嵌入的 normalizer；source checkpoint 提供初始化权重。独立 `normalization.json`、`nuplan_train.json` 仍留在仓库，但不是本次 train/evaluate 入口直接读取的实验输入。

## 独立 pair mining

`datasets/pair/run_pair_mining.py` 从 `<data-root>/Data_<embodiment>/*/pose_lcam_front.txt` 读取两组 TartanGround pose，按配置抽取局部窗口并匹配，写出 `candidates.json`、`candidates.csv` 和抽样可视化。该工具是可选的数据准备步骤，产物不自动进入缓存、导航 manifests 或全量微调复现训练。

可用命令与实际 argparse 一致：

```bash
python -m datasets.pair.run_pair_mining \
  --data-root /path/to/TartanGround \
  --config configs/dataset/pair.yaml \
  --output-dir /path/to/pair_output \
  --embodiment-a anymal --embodiment-b diff
```

`configs/dataset/pair.yaml` 提供窗口长度/步长、空间搜索半径、进出阈值、重采样点数、路径距离阈值、路径关系和可视化随机种子。工具会在两个 embodiment 目录中寻找 pose 文件；命令中的根目录必须按实际数据布局设置。当前没有完成外部数据源、候选数量或配对质量核验，因此不能据此声称已经得到可用于科学配对结论的数据。当前训练树也没有 paired-training、identity-classifier 或跨具身 loss 流程。
