# Research-score当前入口

此包名表示研究代码范围，模型输出score字段仍是归一化x0轨迹，不能据此声称因果score。

- data/：目标fixed-arc、源表示桥、预算、occupancy路线、契约。
- model/：共享backbone外的embodiment条件与零初始化residual。
- training/：manifest到缓存Dataset、loss原语、方法边界。
- evaluation/：D024、安全停车、碰撞、最短路与SR选模指标。
- scripts/：单一正式train_transfer/evaluate_navigation/build_navigation_tasks/summarize_navigation入口。
- configs/：独立transfer_primary和当前proxy_ab配置。
- preflight/：按profile隔离的当前输入、协议、资源和输出检查。
- tests/：相关协议与实现回归。

正式训练：scripts/run_transfer_training.sh；测试：scripts/run_transfer_evaluation.sh。Proxy使用相同入口的PROFILE=proxy_ab分支，CPU及218/20增量已通过，GPU阶段待执行。详见根ARCHITECTURE.md、两份协议及唯一执行计划。
