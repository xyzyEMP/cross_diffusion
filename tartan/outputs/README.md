# Output layout

`final/` is the only retained formal experiment. It contains the 75-sample,
five-environment, three-robot CUDA evaluation and keeps raw checkpoint, basic
adapter, terrain-aware, GT, and constant-velocity results together.

Recommended reading order:

1. `final/evaluation_report.html` — styled report for direct viewing.
   `evaluation_report.md` is retained as the editable Markdown source.
2. `final/animations/town_samples_seed_20260822.gif` — 20 semantic-map rollouts.
3. `final/summary_by_robot.csv` and `summary_by_environment.csv` — breakdowns.
4. `final/per_sample_metrics.csv` — full sample-level analysis.
5. `final/predictions/*.npz` — reproducible raw arrays and all model stages.

## Higher-level town-focused benchmark

`town_focused_v2/` is the current recommended quantitative result. Read:

1. `town_focused_v2/final_report/report.html` — concise public-facing final report.
   For external sharing, use `report_standalone.html` or `report_share_package.zip` in the same directory.
2. `town_focused_v2/evaluation_report.html` — combined technical result.
3. `town_focused_v2/statistical_summary.json` — clustered 95% confidence intervals.
4. `town_focused_v2/open_loop/trusted_primary_metrics.csv` — 444 high-coverage primary samples.
5. `town_focused_v2/closed_loop/per_episode_metrics.csv` — 222 matched kinematic closed-loop episodes.
6. `town_focused_v2/visualizations/open_loop/` — five 2×3 static overviews and matching GIFs.
7. `town_focused_v2/visualizations/closed_loop/` — five 2×3 executed-trajectory overviews and matching GIFs.

The historical `final/` directory remains unchanged for comparison.

`cache/` is outside this directory and contains rebuildable elevation and
semantic surfaces. Temporary smoke-test outputs are intentionally not retained.
