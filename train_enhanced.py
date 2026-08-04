"""
增强版癫痫检测训练脚本
整合多尺度注意力BiLSTM模型和原有EEGNet模型

支持的模型:
1. BaseEEGNet - 原有的基础EEGNet模型
2. AttentionEEGNet - 带注意力机制的EEGNet模型  
3. AttentionBiLSTM - 多尺度注意力BiLSTM模型 (新增)

特性:
- 模型选择和比较
- 注意力权重可视化
- 内存优化训练
- 详细的性能分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import time
import logging
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib支持中文显示，避免乱码
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# 导入模型
from ablation_models.BaseEEGNet import EEGNet as BaseEEGNet
from ablation_models.AttentionEEGNet import EEGNet as AttentionEEGNet
from ablation_models.AttentionBiLSTM import AttentionBiLSTM
from ablation_models.DeepConvNet import DeepConvNet
from ablation_models.ShallowConvNet import ShallowConvNet
from ablation_models.TCFormer import SimplifiedTCFormer as TCFormer

# --- 1. 配置解析 ---
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='增强版癫痫检测训练脚本')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='AttentionBiLSTM',
                       choices=['BaseEEGNet', 'AttentionEEGNet', 'AttentionBiLSTM', 
                               'DeepConvNet', 'ShallowConvNet', 'TCFormer'],
                       help='选择要训练的模型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'], help='设备选择')
    
    # 数据参数
    parser.add_argument('--chans', type=int, default=23, help='EEG通道数')
    parser.add_argument('--samples', type=int, default=512, help='时间点数')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数')
    
    # 注意力模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--attention_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout比例')
    
    # 其他选项
    parser.add_argument('--save_attention', action='store_true', 
                       help='保存注意力权重用于可视化')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                       help='输出目录')
    
    return parser.parse_args()

# --- 2. 日志配置 ---
def setup_logging(output_dir, model_name):
    """设置日志记录"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/logs', exist_ok=True)
    
    log_filename = f'{output_dir}/logs/{model_name}_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

# --- 3. 设备选择 ---
def get_device(device_preference='auto'):
    """智能设备选择"""
    if device_preference != 'auto':
        return torch.device(device_preference)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 使用 Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 使用 CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU")
    
    return device

# --- 4. 模型创建 ---
def create_model(model_name, args):
    """根据参数创建模型"""
    if model_name == 'BaseEEGNet':
        model = BaseEEGNet(
            nb_classes=args.num_classes,
            Chans=args.chans,
            Samples=args.samples,
            dropoutRate=args.dropout
        )
    elif model_name == 'AttentionEEGNet':
        model = AttentionEEGNet(
            nb_classes=args.num_classes,
            Chans=args.chans,
            Samples=args.samples,
            dropoutRate=args.dropout
        )
    elif model_name == 'AttentionBiLSTM':
        model = AttentionBiLSTM(
            nb_classes=args.num_classes,
            Chans=args.chans,
            Samples=args.samples,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_attention=True,
            attention_heads=args.attention_heads
        )
    elif model_name == 'DeepConvNet':
        model = DeepConvNet(
            nb_classes=args.num_classes,
            Chans=args.chans,
            Samples=args.samples,
            dropoutRate=args.dropout
        )
    elif model_name == 'ShallowConvNet':
        model = ShallowConvNet(
            nb_classes=args.num_classes,
            Chans=args.chans,
            Samples=args.samples,
            dropoutRate=args.dropout
        )
    elif model_name == 'TCFormer':
        model = TCFormer(
            nb_classes=args.num_classes,
            Chans=args.chans,
            Samples=args.samples,
            temp_kernels=(16, 32, 64),
            F1=16,
            D=2,
            d_model=64,
            num_heads=8,
            num_layers=4,
            tcn_channels=32,
            tcn_layers=2,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"未知的模型类型: {model_name}")
    
    return model

# --- 5. 数据加载 ---
def load_data():
    """加载训练和测试数据"""
    logger = logging.getLogger(__name__)
    
    try:
        # 加载预处理好的数据
        X_train = np.load('data/X_train_balanced.npy')
        y_train = np.load('data/y_train_balanced.npy')
        X_test = np.load('data/X_test_balanced.npy')
        y_test = np.load('data/y_test_balanced.npy')
        
        logger.info(f"数据加载成功:")
        logger.info(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
        logger.info(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
        logger.info(f"训练集标签分布: {np.bincount(y_train)}")
        logger.info(f"测试集标签分布: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test
        
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        logger.error("请先运行 prepare_data_balanced_memory.py 生成训练数据")
        raise

# --- 6. 训练函数 ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, args):
    """训练模型"""
    logger = logging.getLogger(__name__)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # 打印进度
        logger.info(f'Epoch [{epoch+1}/{epochs}] - '
                   f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                   f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"已加载最佳模型 (验证准确率: {best_val_acc:.2f}%)")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

# --- 7. 评估函数 ---
def evaluate_model(model, test_loader, device, args):
    """评估模型性能"""
    logger = logging.getLogger(__name__)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    logger.info("=== 测试集性能评估 ===")
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1分数: {f1:.4f}")
    logger.info(f"混淆矩阵:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }

# --- 8. 可视化函数 ---
def plot_training_history(history, output_dir, model_name):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 损失曲线
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 损失分布
    ax3.hist(history['train_losses'], bins=20, alpha=0.7, label='Train Loss Dist')
    ax3.set_title('Training Loss Distribution')
    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 准确率分布
    ax4.hist(history['val_accuracies'], bins=20, alpha=0.7, label='Val Acc Dist')
    ax4.set_title('Validation Accuracy Distribution')
    ax4.set_xlabel('Accuracy (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, output_dir, model_name):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'],
                annot_kws={'size': 16})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- 9. 主函数 ---
def main():
    """主训练流程"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    log_file = setup_logging(args.output_dir, args.model)
    logger = logging.getLogger(__name__)
    
    logger.info("=== 增强版癫痫检测训练开始 ===")
    logger.info(f"选择的模型: {args.model}")
    logger.info(f"训练参数: {vars(args)}")
    
    # 设置设备
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    
    # 数据转换为PyTorch格式
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建验证集 (从训练集分割20%)
    val_size = len(train_dataset) // 5
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = create_model(args.model, args)
    model = model.to(device)
    
    # 打印模型信息
    logger.info(f"模型结构:\n{model}")
    
    if hasattr(model, 'get_model_size'):
        size_info = model.get_model_size()
        logger.info(f"模型参数数: {size_info['total_params']:,}")
        logger.info(f"模型大小: {size_info['size_mb']:.2f} MB")
    
    # 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练模型
    logger.info("开始训练...")
    start_time = time.time()
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         device, args.epochs, args)
    
    training_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 评估模型
    logger.info("开始评估...")
    results = evaluate_model(model, test_loader, device, args)
    
    # 保存结果
    os.makedirs(f'{args.output_dir}/models', exist_ok=True)
    torch.save(model.state_dict(), f'{args.output_dir}/models/{args.model}_best.pth')
    logger.info(f"模型已保存到: {args.output_dir}/models/{args.model}_best.pth")
    
    # 绘制图表
    plot_training_history(history, args.output_dir, args.model)
    plot_confusion_matrix(results['confusion_matrix'], args.output_dir, args.model)
    
    # 保存注意力权重 (如果支持)
    if args.save_attention and hasattr(model, 'get_attention_weights'):
        logger.info("保存注意力权重...")
        sample_data = X_test_tensor[:5].to(device)  # 取5个样本
        attention_weights = model.get_attention_weights(sample_data)
        if attention_weights:
            np.save(f'{args.output_dir}/{args.model}_attention_weights.npy', attention_weights)
            logger.info("注意力权重已保存")
    
    logger.info("=== 训练完成 ===")
    logger.info(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
    logger.info(f"测试准确率: {results['accuracy']:.4f}")
    logger.info(f"日志文件: {log_file}")

if __name__ == "__main__":
    main()