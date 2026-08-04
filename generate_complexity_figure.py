"""
重新生成 paper_materials/model_complexity_analysis.png
与 outputs/ 图片保持一致的大字体风格
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 全局字体设置（与 generate_predicted_results.py 保持一致）──
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 26
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 28
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 24
rcParams['figure.titlesize'] = 36
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6

# ── 数据（来自 model_complexity_comparison.csv）──
MODELS = ['BaseEEGNet', 'AttentionEEGNet', 'AttentionBiLSTM',
          'DeepConvNet', 'ShallowConvNet', 'TCFormer']

PARAMS = [25000, 35362, 1128194, 278277, 40122, 227858]
SIZES  = [0.100, 0.135, 4.304, 1.062, 0.153, 0.869]   # MB

# 颜色
BAR_BLUE  = '#90CAF9'
BAR_PINK  = '#EF9A9A'
HIGHLIGHT = '#EF5350'   # AttentionEEGNet 高亮
OTHER     = '#A5D6A7'   # 其他模型

PIE_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

XTICK_ROT = 30

def fmt_param(v):
    """格式化参数量标注"""
    if v >= 1_000_000:
        return f'{v/1e6:.2f}M'
    if v >= 1_000:
        return f'{v/1e3:.1f}K'
    return str(v)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Model Complexity Comparison', fontsize=30, fontweight='bold', y=0.99)

gs = fig.add_gridspec(2, 2, hspace=0.65, wspace=0.35)

x = np.arange(len(MODELS))
bar_w = 0.6

# ── 子图1：总参数量 ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(x, PARAMS, bar_w, color=BAR_BLUE, edgecolor='white', linewidth=0.8)
ax1.set_title('Total Parameters', fontsize=22, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(MODELS, rotation=XTICK_ROT, ha='right', fontsize=24)
ax1.set_ylabel('Number of Parameters', fontsize=18)
ax1.tick_params(axis='y', labelsize=15)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f'{v/1e6:.1f}M' if v >= 1e6 else f'{int(v/1e3)}K' if v >= 1e3 else str(int(v)))
)
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_ylim(0, max(PARAMS) * 1.18)
for bar, val in zip(bars1, PARAMS):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(PARAMS) * 0.01,
             fmt_param(val), ha='center', va='bottom', fontsize=22, fontweight='bold')

# ── 子图2：模型大小 ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(x, SIZES, bar_w, color=BAR_PINK, edgecolor='white', linewidth=0.8)
ax2.set_title('Model Size', fontsize=22, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(MODELS, rotation=XTICK_ROT, ha='right', fontsize=24)
ax2.set_ylabel('Size (MB)', fontsize=18)
ax2.tick_params(axis='y', labelsize=15)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim(0, max(SIZES) * 1.18)
for bar, val in zip(bars2, SIZES):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(SIZES) * 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

# ── 子图3：参数效率（AttentionEEGNet 高亮）──
ax3 = fig.add_subplot(gs[1, 0])
colors3 = [HIGHLIGHT if m == 'AttentionEEGNet' else OTHER for m in MODELS]
bars3 = ax3.bar(x, PARAMS, bar_w, color=colors3, edgecolor='white', linewidth=0.8)
ax3.set_title('Parameter Efficiency  (AttentionEEGNet Highlighted)',
              fontsize=19, fontweight='bold', color=HIGHLIGHT, pad=12)
ax3.set_xticks(x)
ax3.set_xticklabels(MODELS, rotation=XTICK_ROT, ha='right', fontsize=24)
ax3.set_ylabel('Number of Parameters', fontsize=18)
ax3.tick_params(axis='y', labelsize=15)
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f'{v/1e6:.1f}M' if v >= 1e6 else f'{int(v/1e3)}K' if v >= 1e3 else str(int(v)))
)
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_ylim(0, max(PARAMS) * 1.18)
for bar, val in zip(bars3, PARAMS):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(PARAMS) * 0.01,
             fmt_param(val), ha='center', va='bottom', fontsize=22, fontweight='bold')

# ── 子图4：架构分布饼图 ──────────────────────
ax4 = fig.add_subplot(gs[1, 1])
arch_labels = ['CNN-based', 'Transformer-based', 'LSTM-based', 'Attention-based']
arch_sizes  = [3, 1, 1, 1]   # BaseEEGNet/DeepConvNet/ShallowConvNet / TCFormer / AttentionBiLSTM / AttentionEEGNet
wedges, texts, autotexts = ax4.pie(
    arch_sizes,
    labels=arch_labels,
    colors=PIE_COLORS,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 16},
    pctdistance=0.65,
    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
)
for at in autotexts:
    at.set_fontsize(15)
    at.set_fontweight('bold')
for t in texts:
    t.set_fontsize(16)
ax4.set_title('Model Architecture Distribution', fontsize=22, fontweight='bold', pad=14)

out_path = 'paper_materials/model_complexity_analysis.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')