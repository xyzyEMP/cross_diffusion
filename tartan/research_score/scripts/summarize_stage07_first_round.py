from pathlib import Path
import csv, json

root = Path('/tj-share/cross_diffusion_workdir/research_score_v1_2/05_transfer/seed11_first_round')
rows = []
for path in root.glob('*/metrics.json'):
    d = json.loads(path.read_text())
    rows.append({
        'method': d['method'], 'budget_pct': d['budget'], 'seed': d['seed'],
        'target_windows': d['target_samples'], 'best_val_denoising_loss': d['best_validation_loss'],
        'seconds': d['seconds'], 'trainable_parameters': d['trainable_parameters'],
    })
rows.sort(key=lambda x: (x['method'], x['budget_pct']))
with (root / 'first_round_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

methods = ['pretrain_finetune','pretrain_adapter','joint_train','emb_cond_diffusion']
by = {(r['method'], r['budget_pct']): r for r in rows}
lines = [
    '# Stage 07 第一轮训练筛查', '',
    '- 配置：5 种方法 × 1%/10%/100% ANYmal 预算 × seed 11，共 15 次。',
    '- 训练：每次 300 step，AMP；固定 32 个验证窗口，每 100 step 验证并保存最佳模型。',
    '- 实际预算：由于按完整 episode 划分且不能拆分，1%/10%/100% 分别为 128/256/2048 个训练窗口。',
    '- 本轮用途：检查训练能否完成、初步趋势和耗时。下表的 loss 不是最终导航指标，不能替代 SR、碰撞率和 SPL。', '',
    '| 方法 | 1% loss | 10% loss | 100% loss | 1% 时间(s) | 10% 时间(s) | 100% 时间(s) |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for m in methods:
    r1,r10,r100=(by[(m,b)] for b in (1,10,100))
    lines.append(f"| {m} | {r1['best_val_denoising_loss']:.4f} | {r10['best_val_denoising_loss']:.4f} | {r100['best_val_denoising_loss']:.4f} | {r1['seconds']:.1f} | {r10['seconds']:.1f} | {r100['seconds']:.1f} |")
lines += ['', '## 初步观察', '',
    '- 15/15 均完成，checkpoint、逐步日志和 metrics 均已保留。',
    '- 预训练后全量微调在 10% 与 100% 上得到本轮最低验证 loss；target-only 随预算增加持续改善。',
    '- adapter 仅训练约 9.8k 参数，但本轮 loss 明显较高，需要在最终导航指标上决定是否保留或调整容量。',
    '- joint 与 embodiment-conditioned 方法可稳定混合读取原生 nuPlan Car 和 Tartan ANYmal 数据；单 seed 下没有稳定证明优于普通微调。',
    '- 下一步必须用统一 goal-conditioned 闭环 evaluator 计算 SR、碰撞率和 SPL，再决定正式多 seed 训练；不能按当前 loss 直接下迁移结论。',
]
(root / 'first_round_summary.md').write_text('\n'.join(lines) + '\n')
print(root / 'first_round_summary.md')
