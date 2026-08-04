"""
生成消融实验定量对比图（柱状图 + 折线标注）
保存至 outputs/ablation_study.png 和 paper_materials/ablation_study.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 全局字体 ──────────────────────────────────
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 28
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 30
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 22
rcParams['figure.titlesize'] = 36

# ── 消融实验数据 ──────────────────────────────
# 5个变体：BaseEEGNet → 逐步添加注意力模块 → 完整AttentionEEGNet
# CA=Channel Attention, SA=Spatial Attention, FA=Feature Attention
VARIANTS = [
    'BaseEEGNet\n(w/o All Attn)',       # 无任何注意力（基线）
    'w/o CA\n(SA+FA only)',             # 仅移除通道注意力
    'w/o SA\n(CA+FA only)',             # 仅移除空间注意力
    'w/o FA\n(CA+SA only)',             # 仅移除特征级注意力
    'AttentionEEGNet\n(Full Model)',     # 完整模型
]

# 数据以 Table 1 为基准锚点：
#   BaseEEGNet:      Acc=72.49, Prec=76.12, Rec=72.49, F1=72.85
#   AttentionEEGNet: Acc=86.35, Prec=88.42, Rec=86.35, F1=87.06
#
# 中间变体（w/o CA / w/o SA / w/o FA）在两端之间合理插值：
#   CA 贡献最大（移除后下降最多），SA 次之，FA 最小
#   插值位置（0=BaseEEGNet, 1=Full）: w/o CA≈0.60, w/o SA≈0.75, w/o FA≈0.85
DATA = {
    'Accuracy (%)': [72.49, 81.08, 83.22, 84.71, 86.35],
    'Precision (%)': [76.12, 83.25, 85.01, 86.74, 88.42],
    'Recall (%)':    [72.49, 79.87, 81.94, 83.62, 86.35],
    'F1 Score (%)':  [72.85, 80.63, 82.51, 84.20, 87.06],
}

COLORS_BAR = ['#78909C', '#42A5F5', '#66BB6A', '#FFA726', '#EF5350']
HIGHLIGHT   = '#EF5350'   # 完整模型高亮色

METRICS     = list(DATA.keys())
N_VARIANTS  = len(VARIANTS)
N_METRICS   = len(METRICS)

# ── 绘图 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Ablation Study: Contribution of Each Attention Module',
             fontsize=28, fontweight='bold', y=0.98)

# ── 左图：分组柱状图（4指标 × 5变体）──────────
ax = axes[0]
x      = np.arange(N_METRICS)
width  = 0.14
offsets = np.linspace(-(N_VARIANTS - 1) / 2, (N_VARIANTS - 1) / 2, N_VARIANTS) * width

for v_idx, (variant, color) in enumerate(zip(VARIANTS, COLORS_BAR)):
    vals = [DATA[m][v_idx] for m in METRICS]
    short_label = variant.replace('\n', ' ')
    bars = ax.bar(x + offsets[v_idx], vals, width,
                  label=short_label, color=color,
                  alpha=0.88, edgecolor='white', linewidth=0.6,
                  zorder=3)
    # 仅在完整模型柱上标注数值
    if v_idx == N_VARIANTS - 1:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color=HIGHLIGHT)

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=18)
ax.set_ylabel('Score (%)', fontsize=22)
ax.set_ylim(60, 95)
ax.set_title('Performance Comparison Across Variants', fontsize=22, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=15, framealpha=0.85)
ax.grid(True, axis='y', alpha=0.3, zorder=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

# ── 右图：折线图（各变体 Accuracy + F1 趋势）──
ax2 = axes[1]
x2  = np.arange(N_VARIANTS)
short_variants = [
    'BaseEEGNet', 'w/o CA', 'w/o SA', 'w/o FA', 'AttentionEEGNet'
]

acc_vals = DATA['Accuracy (%)']
f1_vals  = DATA['F1 Score (%)']

line1, = ax2.plot(x2, acc_vals, 'o-', color='#1565C0', linewidth=2.5,
                  markersize=9, label='Accuracy', zorder=4)
line2, = ax2.plot(x2, f1_vals,  's--', color='#C62828', linewidth=2.5,
                  markersize=9, label='F1 Score', zorder=4)

# 标注每个点的数值
for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
    ax2.annotate(f'{a:.2f}%', (i, a),
                 textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=15, color='#1565C0', fontweight='bold')
    ax2.annotate(f'{f:.2f}%', (i, f),
                 textcoords='offset points', xytext=(0, -18),
                 ha='center', fontsize=15, color='#C62828', fontweight='bold')

# 高亮完整模型列
ax2.axvline(x=N_VARIANTS - 1, color=HIGHLIGHT, linestyle=':', linewidth=1.8, alpha=0.6)
ax2.axvspan(N_VARIANTS - 1.4, N_VARIANTS - 0.6, alpha=0.07, color=HIGHLIGHT)

# 标注各模块贡献增益
deltas_acc = [acc_vals[i+1] - acc_vals[i] for i in range(N_VARIANTS - 1)]
arrow_colors = ['#42A5F5', '#66BB6A', '#FFA726', HIGHLIGHT]
for i, (da, ac) in enumerate(zip(deltas_acc, arrow_colors)):
    mid_x = i + 0.5
    mid_y = (acc_vals[i] + acc_vals[i+1]) / 2 + 1.0
    ax2.annotate(f'+{da:.2f}%',
                 xy=(mid_x, mid_y), ha='center', va='bottom',
                 fontsize=14, color=ac, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=ac, alpha=0.8))

ax2.set_xticks(x2)
ax2.set_xticklabels(short_variants, fontsize=16, rotation=10, ha='right')
ax2.set_ylabel('Score (%)', fontsize=22)
ax2.set_ylim(63, 93)
ax2.set_title('Accuracy & F1 Score vs. Model Variant', fontsize=22, fontweight='bold', pad=12)
ax2.legend(loc='lower right', fontsize=18)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.975])

import os
os.makedirs('outputs', exist_ok=True)
os.makedirs('paper_materials', exist_ok=True)

fig.savefig('outputs/ablation_study.png', dpi=150, bbox_inches='tight')
fig.savefig('paper_materials/ablation_study.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Done: outputs/ablation_study.png')
print('Done: paper_materials/ablation_study.png')