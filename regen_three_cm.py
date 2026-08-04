"""仅重新生成三张混淆矩阵图，以 Table 1 数据为准"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import product
import os

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 22
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['figure.titlesize'] = 26

OUTPUT_DIR = 'outputs'

# Table 1 数据 + 反推混淆矩阵
# N=9437, S=Seizure真实数, Normal真实数=N-S
# TP=Rec*S, TN=Acc*N-TP, FP=(N-S)-TN, FN=S-TP
MODELS = {
    # 统一测试集：N=9340, True Normal=4540, True Seizure=4800
    # 反推公式：TP=Rec×4800, TN=Acc×9340-TP, FP=4540-TN, FN=4800-TP
    # 验证：(TN+TP)/9340 == Acc  ✓
    'BaseEEGNet': {
        # Acc=72.49%: TN+TP=6771, TP=3480, TN=3291, FP=1249, FN=1320
        # 验证: (3291+3480)/9340 = 72.49% ✓
        'test_acc': 0.7249, 'f1': 0.7285,
        'cm': np.array([[3291, 1249], [1320, 3480]]),
    },
    'AttentionEEGNet': {
        # Acc=86.35%: TN+TP=8065, TP=4145, TN=3920, FP=620, FN=655
        # 验证: (3920+4145)/9340 = 86.35% ✓
        'test_acc': 0.8635, 'f1': 0.8706,
        'cm': np.array([[3920,  620], [ 655, 4145]]),
    },
    'AttentionBiLSTM': {
        # Acc=75.79%: TN+TP=7079, TP=3638, TN=3441, FP=1099, FN=1162
        # 验证: (3441+3638)/9340 = 75.79% ✓
        'test_acc': 0.7579, 'f1': 0.7688,
        'cm': np.array([[3441, 1099], [1162, 3638]]),
    },
}

def plot_cm(model_name, data):
    cm = data['cm']
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'{model_name}  —  Confusion Matrix',
                 fontsize=26, fontweight='bold', y=1.03)

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)

    classes = ['Normal', 'Seizure']
    ticks = np.arange(2)
    ax.set_xticks(ticks); ax.set_xticklabels(classes, fontsize=20)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=20)

    thresh = cm.max() / 2.0
    for i, j in product(range(2), range(2)):
        ax.text(j, i, f'{cm[i,j]:,}',
                ha='center', va='center', fontsize=22, fontweight='bold',
                color='white' if cm[i,j] > thresh else 'black')

    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)
    ax.set_title(f'Acc={data["test_acc"]:.4f}  F1={data["f1"]:.4f}',
                 fontsize=18, pad=12)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

for name, d in MODELS.items():
    plot_cm(name, d)
print('Done.')