# Cross-embodiment local trajectory pairing

该目录从 TartanGround 不同 embodiment 的真实轨迹中提取局部轨迹对。当前配置挖掘：

```text
同一地图 + 不同 embodiment + 同一局部区域
+ 相近入口 + 相近出口 + 相似路线
```

将 `config.yaml` 中的 `path_relation` 改为 `different`，即可切换为“同进同出、路线不同”的候选挖掘。

## 输入数据

数据目录应为：

```text
DATA_ROOT/
├── Data_anymal/<trajectory>/pose_lcam_front.txt
├── Data_diff/<trajectory>/pose_lcam_front.txt
└── Data_omni/<trajectory>/pose_lcam_front.txt
```

每行 pose 为：

```text
x y z qx qy qz qw
```

`load_trajectory()` 将每条 sequence 统一为 `Trajectory`：

- `trajectory_id`：sequence 目录名；
- `map_id`：`Data_<embodiment>` 的父目录名；
- `positions`：`front_lcam` 的 XYZ；
- `quaternions`：原始 XYZW；
- `timestamps`：metadata 存在合法 `time_step` 时使用 `arange(N) * time_step`，否则为 `None`。

时间戳不参与切段或匹配。

## 提取过程

### 1. 按空间距离切局部片段

`extract_local_segments()` 使用连续 XYZ 点之间的三维距离计算累计弧长，不按 frame 数切分。

当前配置：

```yaml
segment_length: 10.0
segment_stride: 5.0
```

因此每个 segment 表示 10 m 实际运动距离，窗口每 5 m 滑动一次。窗口入口、出口和中心在累计弧长上做线性插值；不足 10 m 的尾段不保留。

每个 `LocalSegment` 保存来源 trajectory、原始 frame 范围、XYZ 点、入口、出口、中心和三维路径长度。

### 2. 建立局部空间索引

匹配阶段使用 XY，避免不同机器人相机安装高度造成系统性 Z 偏差。

待搜索 embodiment 的 segment 按中心 XY 放入均匀网格；网格尺寸等于 `region_search_radius`。每个查询 segment 只检查自身网格及相邻八个网格，并再次进行精确中心距离判断，避免全量两两比较。

候选必须满足：

```text
map_id 相同
embodiment 不同
XY center distance <= region_search_radius
XY entry distance < entry_threshold
XY exit distance < exit_threshold
```

### 3. 比较路线

两个 segment 的 XY 路径先按二维弧长均匀重采样为 `resample_points` 个点。连续重复位置会先被去除。

计算指标：

- `mean_path_distance`：对应重采样点距离的平均值；
- `max_path_distance`：对应重采样点距离的最大值；
- `chamfer_distance`：双向最近点距离均值的平均；
- `path_length_ratio`：两个三维 segment 长度之比。

`path_relation: similar` 时，只有 mean 和 max 均严格小于阈值才保留；`path_relation: different` 时，只有两者均严格大于阈值才保留。

`spatial_overlap` 是两条重采样路径在 `region_search_radius` 内的对称最近点覆盖率。目前仅写入结果用于诊断，不参与候选过滤。

## 当前配置

```yaml
segment_length: 10.0
segment_stride: 5.0
region_search_radius: 3.0
entry_threshold: 1.0
exit_threshold: 1.0
resample_points: 64
mean_path_distance_threshold: 1.0
max_path_distance_threshold: 2.0
path_relation: similar
visualization_count: 50
random_seed: 0
```

这些值是首轮人工检查参数，不应视为最终数据定义。

## 输出

每次运行生成：

```text
OUTPUT_DIR/
├── candidates.json
├── candidates.csv
└── visualizations/
    ├── 000.png
    └── ...
```

JSON 和 CSV 保存 pair、map、trajectory、segment、embodiment、入口/出口、中心/入口/出口距离、路线指标及路径长度。CSV 中的坐标数组使用 JSON 字符串表示。

可视化为 XY top-down 图，显示两条局部路线及各自入口和出口；标题包含 map、embodiment、入口/出口距离、mean distance 和 Chamfer distance。最多按固定随机种子输出 50 张。

## 运行

测试：

```bash
python -m unittest discover -s pair -p 'test_*.py' -v
```

单组运行：

```bash
python -m pair.run_pair_mining \
  --data-root /tj-share/tartanground/ModularNeighborhood \
  --config pair/config.yaml \
  --output-dir OUTPUT_DIR \
  --embodiment-a anymal \
  --embodiment-b diff
```

三组配对分别使用：

```text
anymal diff
anymal omni
diff omni
```

## 当前全量结果

输入包含 24 条 anymal、5 条 diff 和 6 条 omni trajectory。当前 `similar` 配置的结果位于：

```text
/tj-share/tartanground/cross_diffusion_workdir/pair/
├── anymal-diff/   # 9 candidates, 9 PNGs
├── anymal-omni/   # 41 candidates, 41 PNGs
└── diff-omni/     # 70 candidates, 50 sampled PNGs
```

每个 candidate 是来自两条完整 trajectory 的两个局部 segment，不是整条 trajectory 的直接比较。

## 文件说明

- `trajectory.py`：统一读取 trajectory，并按三维弧长提取局部 segment；
- `matching.py`：空间索引、局部区域判断、重采样、路线指标和候选过滤；
- `visualization.py`：XY pair 可视化；
- `run_pair_mining.py`：命令行入口及 JSON/CSV/PNG 输出；
- `config.yaml`：所有长度、距离、模式和可视化参数；
- `test_trajectory.py`、`test_matching.py`：最小可运行验证。
