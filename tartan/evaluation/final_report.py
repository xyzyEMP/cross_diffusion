"""Generate the concise, public-facing final report and its core figure."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
import re
import zipfile

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from tartan.viz.report_html import render_report


def _write_standalone_html(html_path: Path, output_path: Path) -> None:
    """Inline local report images so the resulting HTML can be sent alone."""
    html = html_path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        source = match.group(1)
        if source.startswith(("data:", "http://", "https://")):
            return match.group(0)
        asset = (html_path.parent / source).resolve()
        if not asset.exists():
            return match.group(0)
        mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{encoded}"'

    html = re.sub(r'src="([^"]+)"', replace, html)
    output_path.write_text(html, encoding="utf-8")


def _write_share_package(root: Path, destination: Path) -> None:
    files = [
        root / "final_report/report.html",
        root / "final_report/report.md",
        root / "final_report/core_results.png",
        root / "evaluation_report.html",
        root / "evaluation_report.md",
        root / "statistical_summary.json",
        root / "open_loop/trusted_primary_metrics.csv",
        root / "closed_loop/per_episode_metrics.csv",
        *sorted((root / "visualizations/open_loop").glob("*")),
        *sorted((root / "visualizations/closed_loop").glob("*")),
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="tartan/outputs/town_focused_v2")
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    destination = root / "final_report"
    destination.mkdir(parents=True, exist_ok=True)

    open_frame = pd.read_csv(root / "open_loop/trusted_primary_metrics.csv")
    closed = pd.read_csv(root / "closed_loop/per_episode_metrics.csv")
    statistics = json.loads((root / "statistical_summary.json").read_text(encoding="utf-8"))
    om = open_frame.mean(numeric_only=True)
    cm = closed.mean(numeric_only=True)

    cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(cjk_font)
    cjk_family = font_manager.FontProperties(fname=cjk_font).get_name()
    plt.rcParams.update({"font.family": cjk_family, "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    colors = ["#9ca3af", "#60a5fa", "#1d4e89", "#e6a23c"]
    methods = ["原始模型", "基础适配", "地形适配", "匀速基线"]
    ade = [om.raw_model_ade_m, om.basic_model_ade_m, om.model_ade_m, om.baseline_ade_m]
    fde = [om.raw_model_fde_m, om.basic_model_fde_m, om.model_fde_m, om.baseline_fde_m]
    x = np.arange(4)
    width = 0.36
    axes[0].bar(x - width / 2, ade, width, label="ADE", color=colors, alpha=.75)
    axes[0].bar(x + width / 2, fde, width, label="FDE", color=colors, hatch="///", alpha=.95)
    axes[0].set_xticks(x, methods, rotation=18, ha="right")
    axes[0].set_ylabel("误差（米，越低越好）")
    axes[0].set_title("开环轨迹误差")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].bar([0, 1], [cm.success * 100, cm.cv_success * 100], color=["#1d4e89", "#e6a23c"])
    axes[1].set_xticks([0, 1], ["滚动规划", "匀速基线"])
    axes[1].set_ylabel("严格成功率（%）")
    axes[1].set_title("运动学闭环成功率")
    for index, value in enumerate([cm.success * 100, cm.cv_success * 100]):
        axes[1].text(index, value + 0.8, f"{value:.1f}%", ha="center")
    axes[1].set_ylim(0, max(cm.success, cm.cv_success) * 125)

    robot = closed.groupby("robot_type")[["success", "cv_success"]].mean().reindex(["anymal", "diff", "omni"])
    x = np.arange(3)
    axes[2].bar(x - width / 2, robot.success * 100, width, label="滚动规划", color="#1d4e89")
    axes[2].bar(x + width / 2, robot.cv_success * 100, width, label="匀速基线", color="#e6a23c")
    axes[2].set_xticks(x, ["四足 ANYmal", "差速轮式", "全向轮式"])
    axes[2].set_ylabel("严格成功率（%）")
    axes[2].set_title("不同机器人闭环表现")
    axes[2].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
    figure_path = destination / "core_results.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    ci = statistics["cluster_bootstrap_95ci"]
    report = f"""# Diffusion-Planner 在 TartanGround 上的可运行性与迁移基线

## 任务

本项目测试一个原本在 nuPlan 城市道路数据上训练的 Diffusion-Planner，能否在未重新训练模型的情况下，为 TartanGround 中的轮式和四足机器人生成合理轨迹。ModernCityDowntown 与 OldTownFall 是主测试环境，其余自然环境用于辅助泛化检查。

## 方法

模型权重保持不变。我们将 TartanGround 位姿转换为模型需要的历史状态和路线条件，并加入速度尺度适配、轮式运动学约束、地形代价筛选、坡度/台阶/支撑检查、危险前减速截停和轨迹平滑。

测试分为两级：

- **开环测试**：一次预测未来 8 秒，比较预测轨迹与记录轨迹。
- **运动学闭环测试**：每执行 1 秒重新规划，并加入位置、航向和跟踪扰动，检查能否持续沿路线行进。

数据共 46 条轨迹、7.52 GiB。开环包含 552 个样本，其中核心结果采用主环境内地图覆盖率不低于 95% 的 444 个样本；闭环包含 222 个 episode。

## 对比方法与指标

报告中的“四个模型”更准确地说是四种对比方法，并不是四套分别训练的神经网络：

- **原始模型（Raw checkpoint）**：直接使用 nuPlan 预训练权重，只转换输入格式，不修正速度尺度或机器人约束，用来观察最原始的跨域表现。
- **基础适配（Basic adapter）**：不修改模型权重，根据 TartanGround 最近的运动速度重新调整轨迹时间尺度，并限制速度、加速度和转向，解决“预测明显过快”等问题。
- **地形适配（Terrain-aware）**：在基础适配上加入地形引导、候选轨迹重排序、轮式运动学投影、地形减速、安全截停和轨迹平滑，是当前完整方法。
- **匀速基线（Constant velocity）**：不使用 Diffusion-Planner，只假设机器人保持最近观测到的速度和转向趋势前进。它代表一个简单但必要的参考方法。

核心指标可以这样理解：

- **ADE**：未来 8 秒内，每个时刻预测位置与记录轨迹之间距离的平均值；越小越好。
- **FDE**：第 8 秒末端预测位置与记录终点之间的距离；越小越好。
- **路线偏离**：轨迹上的点到记录路线的平均最近距离；越小表示越能沿既定走廊行进。
- **严格成功率**：一个闭环 episode 同时满足“最终误差小于 2 米、路线进度超过 80%、没有几何失败”才记为成功，成功 episode 数除以总 episode 数即为成功率。
- **几何失败率**：执行轨迹出现明显地图缺失、支撑不足，或坡度/台阶超限比例过高的 episode 占比。它是静态几何代理，不等于真实碰撞率或四足动力学失败率。

闭环部分主要比较“完整地形适配后的滚动规划”和“匀速基线”。原始模型与基础适配主要用于开环消融，帮助判断每一层适配究竟贡献了什么。

## 核心结果

![开环、闭环与机器人对比](core_results.png)

地形适配后的开环 ADE/FDE 为 **{om.model_ade_m:.2f}/{om.model_fde_m:.2f} 米**，优于匀速基线的 **{om.baseline_ade_m:.2f}/{om.baseline_fde_m:.2f} 米**，也明显优于未经适配的原始模型。开环 ADE 相对匀速基线平均降低 **{-ci['open_ade_final_minus_cv_m']['estimate']:.2f} 米**，95% 置信区间不跨零。

运动学闭环中，模型严格成功率为 **{cm.success:.1%}**，匀速基线为 **{cm.cv_success:.1%}**；模型 FDE 从基线的 **{cm.cv_fde_m:.2f} 米**降至 **{cm.closed_loop_fde_m:.2f} 米**。成功率优势为 **{ci['closed_success_minus_cv']['estimate'] * 100:.1f} 个百分点**，按轨迹聚类的 95% 置信区间为 **[{ci['closed_success_minus_cv']['ci95_low'] * 100:.1f}, {ci['closed_success_minus_cv']['ci95_high'] * 100:.1f}] 个百分点**。

结果同时暴露了明确限制：四足 ANYmal 成功率为 **{robot.loc['anymal', 'success']:.1%}**，但差速和全向轮式均只有 **{robot.loc['diff', 'success']:.1%}** 和 **{robot.loc['omni', 'success']:.1%}**。因此本阶段最稳妥的结论只是：流程已经跑通，后处理能修正明显的速度和地形问题，但原模型的跨数据集、跨平台表征并没有真正完成对齐，整体迁移效果仍然有限。

## 直观案例

每张图覆盖两个主环境和三类机器人。背景颜色表示道路、植被、建筑和台阶等地物；绿线是记录路线，红线是模型轨迹。五组案例由不同轨迹组成。

### 开环案例

#### 第 1 组
![开环案例第1组](../visualizations/open_loop/open_loop_batch_01_static.png)
![开环动态第1组](../visualizations/open_loop/open_loop_batch_01.gif)

#### 第 2 组
![开环案例第2组](../visualizations/open_loop/open_loop_batch_02_static.png)
![开环动态第2组](../visualizations/open_loop/open_loop_batch_02.gif)

#### 第 3 组
![开环案例第3组](../visualizations/open_loop/open_loop_batch_03_static.png)
![开环动态第3组](../visualizations/open_loop/open_loop_batch_03.gif)

#### 第 4 组
![开环案例第4组](../visualizations/open_loop/open_loop_batch_04_static.png)
![开环动态第4组](../visualizations/open_loop/open_loop_batch_04.gif)

#### 第 5 组
![开环案例第5组](../visualizations/open_loop/open_loop_batch_05_static.png)
![开环动态第5组](../visualizations/open_loop/open_loop_batch_05.gif)

### 闭环案例

闭环红线不是一次性预测，而是每秒重规划后累计得到的实际执行轨迹。五组按各机器人最终误差由低到高排列，便于同时观察容易、一般和困难案例。

#### 第 1 组（较容易）
![闭环案例第1组](../visualizations/closed_loop/closed_loop_batch_01_static.png)
![闭环动态第1组](../visualizations/closed_loop/closed_loop_batch_01.gif)

#### 第 2 组
![闭环案例第2组](../visualizations/closed_loop/closed_loop_batch_02_static.png)
![闭环动态第2组](../visualizations/closed_loop/closed_loop_batch_02.gif)

#### 第 3 组
![闭环案例第3组](../visualizations/closed_loop/closed_loop_batch_03_static.png)
![闭环动态第3组](../visualizations/closed_loop/closed_loop_batch_03.gif)

#### 第 4 组
![闭环案例第4组](../visualizations/closed_loop/closed_loop_batch_04_static.png)
![闭环动态第4组](../visualizations/closed_loop/closed_loop_batch_04.gif)

#### 第 5 组（较困难）
![闭环案例第5组](../visualizations/closed_loop/closed_loop_batch_05_static.png)
![闭环动态第5组](../visualizations/closed_loop/closed_loop_batch_05.gif)

## 结论

> 当前工作建立了一个 zero-shot 迁移基线：Diffusion-Planner 能够在 TartanGround 上完成轨迹输出、地形适配和简化滚动执行，但效果只能视为“跑通”，尚不足以说明已经获得可靠的跨域、跨平台迁移能力。

当前闭环仍是静态地图上的简化运动学仿真，使用记录轨迹提供高层路线；它不包含实时感知、动态障碍、轮胎/足端接触动力学。因此结果不能等同于真实机器人部署性能。

## 下一步

当前系统直接把 TartanGround 信息转换成 nuPlan 风格输入，再依赖平台相关后处理修正输出。这解决了接口问题，却没有保证不同环境、不同机器人对同一任务意图形成一致理解。并且目前高层路线来自记录真值，是 oracle 条件。

下一阶段计划显式引入位于“环境感知”和“具体机器人轨迹”之间的中间表征，表达：

- 要沿哪条可通行走廊前进，以及期望到达的局部目标；
- 期望进度和速度趋势；
- 坡度、台阶、支撑风险等与平台能力相关但不绑定具体控制器的信息。

这个表征应尽量保持**高层意图跨平台一致**，再由轮式、全向和四足各自的平台适配器把同一意图解码成可执行轨迹。训练时可以加入跨平台一致性、轨迹到意图的重建约束和地形增强一致性；少样本阶段优先训练意图编码器、平台适配器或轻量参数，而不是重新训练整个 Diffusion-Planner。任务输出仍保持在轨迹层，不必直接扩展到足端或电机控制。

下一步目标：

> 学习一个显式、地形感知且跨执行平台一致的高层意图表征，并通过少样本对齐，使预训练轨迹规划模型在 TartanGround 上获得稳定优于 zero-shot 基线的迁移性能。

详细数据可查阅：

- [`../open_loop/trusted_primary_metrics.csv`](../open_loop/trusted_primary_metrics.csv)：核心开环样本。
- [`../closed_loop/per_episode_metrics.csv`](../closed_loop/per_episode_metrics.csv)：全部闭环 episode。
- [`../statistical_summary.json`](../statistical_summary.json)：置信区间。
- [`../evaluation_report.html`](../evaluation_report.html)：完整版技术报告。
"""
    markdown_path = destination / "report.md"
    markdown_path.write_text(report, encoding="utf-8")
    html_path = destination / "report.html"
    render_report(markdown_path, html_path)
    _write_standalone_html(html_path, destination / "report_standalone.html")
    _write_share_package(root, destination / "report_share_package.zip")
    print(destination / "report.html")


if __name__ == "__main__":
    main()
