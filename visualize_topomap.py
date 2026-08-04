"""
脑电图拓扑图可视化脚本 - 支持多种模型
用于可视化不同模型学习到的空间特征模式
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import argparse

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 30,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22
})

# 导入模型
sys.path.append(os.path.join(os.path.dirname(__file__), 'ablation_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'advanced_models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'comparison_models'))

# --- 1. 参数设置 ---
CHANS = 23  # EEG通道数 (CHB-MIT)
SAMPLES = 512  # 时间采样点
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# CHB-MIT数据集的23个通道名称
CH_NAMES_23 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'T9', 'T10', 'FC6', 'FC5'
]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EEG拓扑图可视化工具')
    
    parser.add_argument('--model', type=str, default='BaseEEGNet',
                        choices=['BaseEEGNet', 'AttentionEEGNet', 'AttentionBiLSTM', 
                                'DeepConvNet', 'ShallowConvNet', 'TCFormer'],
                        help='选择要可视化的模型')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='指定模型文件路径 (可选，会自动搜索)')
    
    parser.add_argument('--data_path', type=str, default='data',
                        help='数据文件路径')
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    return parser.parse_args()

def create_model(model_name, args=None):
    """根据模型名称创建模型实例"""
    if model_name == 'BaseEEGNet':
        from ablation_models.BaseEEGNet import EEGNet
        return EEGNet(
            nb_classes=2,
            Chans=CHANS,
            Samples=SAMPLES,
            dropoutRate=0.15,
            F1=8, D=2, F2=16
        )
    
    elif model_name == 'AttentionEEGNet':
        from ablation_models.AttentionEEGNet import EEGNet
        return EEGNet(
            nb_classes=2,
            Chans=CHANS,
            Samples=SAMPLES,
            dropoutRate=0.15,
            F1=8, D=2, F2=16
        )
    
    elif model_name == 'AttentionBiLSTM':
        from advanced_models.AttentionBiLSTM import AttentionBiLSTM
        return AttentionBiLSTM(
            nb_classes=2,
            Chans=CHANS,
            Samples=SAMPLES,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            use_attention=True,
            attention_heads=4
        )
    
    elif model_name == 'DeepConvNet':
        from ablation_models.DeepConvNet import DeepConvNet
        return DeepConvNet(
            nb_classes=2,
            Chans=CHANS,
            Samples=SAMPLES,
            dropoutRate=0.5
        )
    
    elif model_name == 'ShallowConvNet':
        from ablation_models.ShallowConvNet import ShallowConvNet
        return ShallowConvNet(
            nb_classes=2,
            Chans=CHANS,
            Samples=SAMPLES,
            dropoutRate=0.5
        )
    
    elif model_name == 'TCFormer':
        from ablation_models.TCFormer import SimplifiedTCFormer
        return SimplifiedTCFormer(
            nb_classes=2,
            Chans=CHANS,
            Samples=SAMPLES,
            temp_kernels=(16, 32, 64),
            F1=16,
            D=2,
            d_model=64,
            num_heads=8,
            num_layers=4,
            tcn_channels=32,
            tcn_layers=2,
            dropout=0.3
        )
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def get_spatial_weights_for_topomap(model, model_name, data_loader):
    """
    从不同模型中提取空间权重用于拓扑图可视化
    
    Args:
        model (nn.Module): 训练好的模型
        model_name (str): 模型名称
        data_loader (DataLoader): 测试数据加载器
        
    Returns:
        np.array: 23个通道的权重分数 (1D数组)
    """
    model.eval()
    
    if model_name in ['BaseEEGNet', 'AttentionEEGNet']:
        return get_eegnet_weights(model, data_loader)
    elif model_name == 'AttentionBiLSTM':
        return get_bilstm_weights(model, data_loader)
    elif model_name in ['DeepConvNet', 'ShallowConvNet']:
        return get_convnet_weights(model, model_name, data_loader)
    elif model_name == 'TCFormer':
        return get_tcformer_weights(model, data_loader)
    else:
        print(f"未知模型类型: {model_name}")
        return np.ones(CHANS) / CHANS

def get_eegnet_weights(model, data_loader):
    """提取EEGNet系列模型的空间权重"""
    try:
        # 获取Block2中深度卷积层的权重
        spatial_conv = None
        for layer in model.block2:
            if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
                spatial_conv = layer
                break
        
        if spatial_conv is None:
            print("未找到深度卷积层，使用默认权重")
            return np.ones(CHANS) / CHANS
            
        spatial_weights = spatial_conv.weight.data.cpu().numpy()
        print(f"空间卷积权重形状: {spatial_weights.shape}")
        
        # 计算每个通道的重要性分数
        if len(spatial_weights.shape) == 4:
            channel_scores = np.mean(np.abs(spatial_weights), axis=(0, 3))
            if channel_scores.shape[1] == CHANS:
                channel_scores = np.mean(channel_scores, axis=0)
            else:
                channel_scores = np.ones(CHANS) / CHANS
        else:
            channel_scores = np.ones(CHANS) / CHANS
        
        # 使用实际数据增强权重计算
        all_activations = []
        
        with torch.no_grad():
            sample_count = 0
            for inputs, labels in data_loader:
                if sample_count >= 10:
                    break
                    
                inputs = inputs.to(DEVICE)
                
                # 获取中间特征图
                if hasattr(model, 'get_feature_maps'):
                    features = model.get_feature_maps(inputs)
                    
                    if 'block2' in features:
                        block2_features = features['block2']
                        activation = torch.mean(torch.abs(block2_features), dim=(0, 3))
                        activation = activation.squeeze().cpu().numpy()
                        
                        if len(activation.shape) == 1 and len(activation) >= CHANS:
                            if len(activation) > CHANS:
                                activation = activation[:CHANS]
                            all_activations.append(activation)
                
                sample_count += 1
        
        # 结合静态权重和动态激活
        if all_activations:
            mean_activation = np.mean(all_activations, axis=0)
            if len(mean_activation) == CHANS:
                final_scores = channel_scores * 0.3 + mean_activation * 0.7
            else:
                final_scores = channel_scores
        else:
            final_scores = channel_scores
        
    except Exception as e:
        print(f"提取EEGNet权重时出错: {e}")
        final_scores = np.ones(CHANS) / CHANS
    
    # 归一化
    final_scores = final_scores / (np.max(np.abs(final_scores)) + 1e-8)
    return final_scores

def get_tcformer_weights(model, data_loader):
    """提取TCFormer模型的权重"""
    try:
        # TCFormer结合了卷积、Transformer和TCN，我们主要关注卷积前端的空间权重
        conv_frontend = model.conv_frontend
        
        # 获取多核卷积块中的空间卷积权重
        spatial_weights_list = []
        
        for spatial_conv_block in conv_frontend.spatial_convs:
            # 获取空间卷积层（深度卷积）
            spatial_conv = spatial_conv_block[0]  # 第一个是Conv2d层
            if hasattr(spatial_conv, 'weight'):
                weights = spatial_conv.weight.data.cpu().numpy()
                if len(weights.shape) == 4 and weights.shape[2] == CHANS:
                    # 对输出通道和其他维度求平均，保留通道维度
                    channel_scores = np.mean(np.abs(weights), axis=(0, 1, 3))
                    spatial_weights_list.append(channel_scores)
        
        # 如果获取到多个空间权重，取平均
        if spatial_weights_list:
            channel_scores = np.mean(spatial_weights_list, axis=0)
        else:
            channel_scores = np.ones(CHANS) / CHANS
        
        # 使用实际数据增强权重计算
        all_activations = []
        
        with torch.no_grad():
            sample_count = 0
            for inputs, labels in data_loader:
                if sample_count >= 10:
                    break
                    
                inputs = inputs.to(DEVICE)
                
                # 获取中间特征图
                if hasattr(model, 'get_feature_maps'):
                    features = model.get_feature_maps(inputs)
                    
                    # 使用卷积特征
                    if 'conv_features' in features:
                        conv_features = features['conv_features']  # [B, channels, T]
                        
                        # 计算激活强度
                        activation = torch.mean(torch.abs(conv_features), dim=(0, 2))  # [channels]
                        activation = activation.cpu().numpy()
                        
                        # 映射到通道数
                        if len(activation) >= CHANS:
                            if len(activation) == CHANS:
                                all_activations.append(activation)
                            else:
                                # 分组平均映射到通道数
                                group_size = len(activation) // CHANS
                                grouped_activation = []
                                for i in range(CHANS):
                                    start_idx = i * group_size
                                    end_idx = start_idx + group_size
                                    if end_idx <= len(activation):
                                        grouped_activation.append(np.mean(activation[start_idx:end_idx]))
                                    else:
                                        grouped_activation.append(activation[start_idx])
                                all_activations.append(np.array(grouped_activation))
                
                sample_count += 1
        
        # 结合静态权重和动态激活
        if all_activations:
            mean_activation = np.mean(all_activations, axis=0)
            if len(mean_activation) == CHANS:
                # 结合静态权重和动态激活
                final_scores = channel_scores * 0.3 + mean_activation * 0.7
            else:
                final_scores = channel_scores
        else:
            final_scores = channel_scores
        
    except Exception as e:
        print(f"提取TCFormer权重时出错: {e}")
        final_scores = np.ones(CHANS) / CHANS
    
    # 归一化
    final_scores = final_scores / (np.max(np.abs(final_scores)) + 1e-8)
    return final_scores

def get_bilstm_weights(model, data_loader):
    """提取BiLSTM模型的注意力权重"""
    try:
        all_attentions = []
        
        with torch.no_grad():
            sample_count = 0
            for inputs, labels in data_loader:
                if sample_count >= 10:
                    break
                    
                inputs = inputs.to(DEVICE)
                
                # BiLSTM模型返回 (output, attention_weights)
                if hasattr(model, 'forward'):
                    try:
                        output, attention_weights = model(inputs)
                        if attention_weights is not None:
                            # attention_weights shape: [B, seq_len] 或 [B, input_dim]
                            att_weights = attention_weights.cpu().numpy()
                            
                            # 如果是序列注意力，需要映射到通道
                            if att_weights.shape[1] == CHANS:
                                all_attentions.append(np.mean(att_weights, axis=0))
                            elif att_weights.shape[1] == SAMPLES:
                                # 时间注意力，需要转换为通道注意力
                                # 这里简化处理，使用均匀分布
                                channel_att = np.ones(CHANS) / CHANS
                                all_attentions.append(channel_att)
                    except:
                        # 如果模型不返回注意力权重，使用均匀分布
                        pass
                
                sample_count += 1
        
        if all_attentions:
            final_scores = np.mean(all_attentions, axis=0)
        else:
            # 如果无法获取注意力权重，使用LSTM权重近似
            final_scores = get_lstm_weight_approximation(model)
        
    except Exception as e:
        print(f"提取BiLSTM权重时出错: {e}")
        final_scores = np.ones(CHANS) / CHANS
    
    # 归一化
    final_scores = final_scores / (np.max(np.abs(final_scores)) + 1e-8)
    return final_scores

def get_tcformer_weights(model, data_loader):
    """提取TCFormer模型的权重"""
    try:
        # TCFormer结合了卷积、Transformer和TCN，我们主要关注卷积前端的空间权重
        conv_frontend = model.conv_frontend
        
        # 获取多核卷积块中的空间卷积权重
        spatial_weights_list = []
        
        for spatial_conv_block in conv_frontend.spatial_convs:
            # 获取空间卷积层（深度卷积）
            spatial_conv = spatial_conv_block[0]  # 第一个是Conv2d层
            if hasattr(spatial_conv, 'weight'):
                weights = spatial_conv.weight.data.cpu().numpy()
                if len(weights.shape) == 4 and weights.shape[2] == CHANS:
                    # 对输出通道和其他维度求平均，保留通道维度
                    channel_scores = np.mean(np.abs(weights), axis=(0, 1, 3))
                    spatial_weights_list.append(channel_scores)
        
        # 如果获取到多个空间权重，取平均
        if spatial_weights_list:
            channel_scores = np.mean(spatial_weights_list, axis=0)
        else:
            channel_scores = np.ones(CHANS) / CHANS
        
        # 使用实际数据增强权重计算
        all_activations = []
        
        with torch.no_grad():
            sample_count = 0
            for inputs, labels in data_loader:
                if sample_count >= 10:
                    break
                    
                inputs = inputs.to(DEVICE)
                
                # 获取中间特征图
                if hasattr(model, 'get_feature_maps'):
                    features = model.get_feature_maps(inputs)
                    
                    # 使用卷积特征
                    if 'conv_features' in features:
                        conv_features = features['conv_features']  # [B, channels, T]
                        
                        # 计算激活强度
                        activation = torch.mean(torch.abs(conv_features), dim=(0, 2))  # [channels]
                        activation = activation.cpu().numpy()
                        
                        # 映射到通道数
                        if len(activation) >= CHANS:
                            if len(activation) == CHANS:
                                all_activations.append(activation)
                            else:
                                # 分组平均映射到通道数
                                group_size = len(activation) // CHANS
                                grouped_activation = []
                                for i in range(CHANS):
                                    start_idx = i * group_size
                                    end_idx = start_idx + group_size
                                    if end_idx <= len(activation):
                                        grouped_activation.append(np.mean(activation[start_idx:end_idx]))
                                    else:
                                        grouped_activation.append(activation[start_idx])
                                all_activations.append(np.array(grouped_activation))
                
                sample_count += 1
        
        # 结合静态权重和动态激活
        if all_activations:
            mean_activation = np.mean(all_activations, axis=0)
            if len(mean_activation) == CHANS:
                # 结合静态权重和动态激活
                final_scores = channel_scores * 0.3 + mean_activation * 0.7
            else:
                final_scores = channel_scores
        else:
            final_scores = channel_scores
        
    except Exception as e:
        print(f"提取TCFormer权重时出错: {e}")
        final_scores = np.ones(CHANS) / CHANS
    
    # 归一化
    final_scores = final_scores / (np.max(np.abs(final_scores)) + 1e-8)
    return final_scores

def get_lstm_weight_approximation(model):
    """从LSTM权重近似计算通道重要性"""
    try:
        # 获取第一层LSTM的输入权重
        if hasattr(model, 'bilstm'):
            lstm_layer = model.bilstm
            weight_ih = lstm_layer.weight_ih_l0.data.cpu().numpy()  # [4*hidden_size, input_size]
            
            # 计算每个输入特征(通道)的权重重要性
            input_importance = np.mean(np.abs(weight_ih), axis=0)  # [input_size]
            
            if len(input_importance) == CHANS:
                return input_importance
            else:
                return np.ones(CHANS) / CHANS
        else:
            return np.ones(CHANS) / CHANS
    except:
        return np.ones(CHANS) / CHANS

def get_convnet_weights(model, model_name, data_loader):
    """提取DeepConvNet和ShallowConvNet的空间权重"""
    try:
        # 获取空间卷积层的权重
        spatial_conv = None
        
        if model_name == 'DeepConvNet':
            # DeepConvNet的第一个block中的空间卷积
            for layer in model.block1:
                if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size[0] == CHANS:
                    spatial_conv = layer
                    break
        elif model_name == 'ShallowConvNet':
            # ShallowConvNet的空间卷积层
            spatial_conv = model.spatial_conv
        
        if spatial_conv is None:
            print(f"未找到{model_name}的空间卷积层，使用默认权重")
            return np.ones(CHANS) / CHANS
            
        spatial_weights = spatial_conv.weight.data.cpu().numpy()
        print(f"{model_name}空间卷积权重形状: {spatial_weights.shape}")
        
        # 计算每个通道的重要性分数
        if len(spatial_weights.shape) == 4:
            # 对输出通道和其他维度求平均，保留通道维度
            if spatial_weights.shape[2] == CHANS:  # 空间维度是通道数
                channel_scores = np.mean(np.abs(spatial_weights), axis=(0, 1, 3))  # 形状: (Chans,)
            else:
                channel_scores = np.ones(CHANS) / CHANS
        else:
            channel_scores = np.ones(CHANS) / CHANS
        
        # 使用实际数据增强权重计算
        all_activations = []
        
        with torch.no_grad():
            sample_count = 0
            for inputs, labels in data_loader:
                if sample_count >= 10:
                    break
                    
                inputs = inputs.to(DEVICE)
                
                # 获取中间特征图
                if hasattr(model, 'get_feature_maps'):
                    features = model.get_feature_maps(inputs)
                    
                    # 根据模型类型选择特征图
                    if model_name == 'DeepConvNet' and 'block1' in features:
                        block_features = features['block1']
                    elif model_name == 'ShallowConvNet' and 'spatial_conv' in features:
                        block_features = features['spatial_conv']
                    else:
                        block_features = None
                    
                    if block_features is not None:
                        # 计算激活强度
                        activation = torch.mean(torch.abs(block_features), dim=(0, 2, 3))  # [out_channels]
                        activation = activation.cpu().numpy()
                        
                        # 如果输出通道数大于输入通道数，需要映射回输入通道
                        if len(activation) >= CHANS:
                            # 简化处理：取前CHANS个或平均分组
                            if len(activation) == CHANS:
                                all_activations.append(activation)
                            else:
                                # 分组平均
                                group_size = len(activation) // CHANS
                                grouped_activation = []
                                for i in range(CHANS):
                                    start_idx = i * group_size
                                    end_idx = start_idx + group_size
                                    grouped_activation.append(np.mean(activation[start_idx:end_idx]))
                                all_activations.append(np.array(grouped_activation))
                
                sample_count += 1
        
        # 结合静态权重和动态激活
        if all_activations:
            mean_activation = np.mean(all_activations, axis=0)
            if len(mean_activation) == CHANS:
                # 结合静态权重和动态激活
                final_scores = channel_scores * 0.4 + mean_activation * 0.6
            else:
                final_scores = channel_scores
        else:
            final_scores = channel_scores
        
    except Exception as e:
        print(f"提取{model_name}权重时出错: {e}")
        final_scores = np.ones(CHANS) / CHANS
    
    # 归一化
    final_scores = final_scores / (np.max(np.abs(final_scores)) + 1e-8)
    return final_scores

def get_tcformer_weights(model, data_loader):
    """提取TCFormer模型的权重"""
    try:
        # TCFormer结合了卷积、Transformer和TCN，我们主要关注卷积前端的空间权重
        conv_frontend = model.conv_frontend
        
        # 获取多核卷积块中的空间卷积权重
        spatial_weights_list = []
        
        for spatial_conv_block in conv_frontend.spatial_convs:
            # 获取空间卷积层（深度卷积）
            spatial_conv = spatial_conv_block[0]  # 第一个是Conv2d层
            if hasattr(spatial_conv, 'weight'):
                weights = spatial_conv.weight.data.cpu().numpy()
                if len(weights.shape) == 4 and weights.shape[2] == CHANS:
                    # 对输出通道和其他维度求平均，保留通道维度
                    channel_scores = np.mean(np.abs(weights), axis=(0, 1, 3))
                    spatial_weights_list.append(channel_scores)
        
        # 如果获取到多个空间权重，取平均
        if spatial_weights_list:
            channel_scores = np.mean(spatial_weights_list, axis=0)
        else:
            channel_scores = np.ones(CHANS) / CHANS
        
        # 使用实际数据增强权重计算
        all_activations = []
        
        with torch.no_grad():
            sample_count = 0
            for inputs, labels in data_loader:
                if sample_count >= 10:
                    break
                    
                inputs = inputs.to(DEVICE)
                
                # 获取中间特征图
                if hasattr(model, 'get_feature_maps'):
                    features = model.get_feature_maps(inputs)
                    
                    # 使用卷积特征
                    if 'conv_features' in features:
                        conv_features = features['conv_features']  # [B, channels, T]
                        
                        # 计算激活强度
                        activation = torch.mean(torch.abs(conv_features), dim=(0, 2))  # [channels]
                        activation = activation.cpu().numpy()
                        
                        # 映射到通道数
                        if len(activation) >= CHANS:
                            if len(activation) == CHANS:
                                all_activations.append(activation)
                            else:
                                # 分组平均映射到通道数
                                group_size = len(activation) // CHANS
                                grouped_activation = []
                                for i in range(CHANS):
                                    start_idx = i * group_size
                                    end_idx = start_idx + group_size
                                    if end_idx <= len(activation):
                                        grouped_activation.append(np.mean(activation[start_idx:end_idx]))
                                    else:
                                        grouped_activation.append(activation[start_idx])
                                all_activations.append(np.array(grouped_activation))
                
                sample_count += 1
        
        # 结合静态权重和动态激活
        if all_activations:
            mean_activation = np.mean(all_activations, axis=0)
            if len(mean_activation) == CHANS:
                # 结合静态权重和动态激活
                final_scores = channel_scores * 0.3 + mean_activation * 0.7
            else:
                final_scores = channel_scores
        else:
            final_scores = channel_scores
        
    except Exception as e:
        print(f"提取TCFormer权重时出错: {e}")
        final_scores = np.ones(CHANS) / CHANS
    
    # 归一化
    final_scores = final_scores / (np.max(np.abs(final_scores)) + 1e-8)
    return final_scores

def find_model_file(model_name, model_path=None):
    """查找模型文件"""
    if model_path and os.path.exists(model_path):
        return model_path
    
    # 自动搜索模型文件
    search_paths = [
        f'outputs/models/{model_name}_best.pth',
        f'{model_name}_best.pth',
        f'outputs/models/best_{model_name.lower()}_model.pth',
        'best_model.pth'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None

def main_visualize():
    """主可视化函数"""
    args = parse_args()
    
    print(f"=== EEG拓扑图可视化 - {args.model} ===")
    
    # 1. 创建模型
    try:
        model = create_model(args.model, args)
        model = model.to(DEVICE)
        print(f"成功创建模型: {args.model}")
    except Exception as e:
        print(f"创建模型失败: {e}")
        return
    
    # 2. 查找并加载模型文件
    model_path = find_model_file(args.model, args.model_path)
    
    if model_path is None:
        print(f"错误: 未找到 {args.model} 的训练好的模型文件")
        print("请确保模型已经训练完成，或使用 --model_path 指定模型文件路径")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 3. 加载测试数据
    data_paths = [
        (f'{args.data_path}/X_test_balanced.npy', f'{args.data_path}/y_test_balanced.npy'),
        (f'{args.data_path}/X_test.npy', f'{args.data_path}/y_test.npy')
    ]
    
    X_test, Y_test = None, None
    for x_path, y_path in data_paths:
        try:
            if os.path.exists(x_path) and os.path.exists(y_path):
                X_test = np.load(x_path)
                Y_test = np.load(y_path)
                print(f"成功加载数据: {x_path}, {y_path}")
                break
        except Exception as e:
            print(f"加载数据失败 {x_path}: {e}")
            continue
    
    if X_test is None or Y_test is None:
        print("错误: 未找到测试数据文件")
        return
    
    # 创建数据加载器
    X_test_tensor = torch.from_numpy(X_test).to(torch.float32)
    Y_test_tensor = torch.from_numpy(Y_test).to(torch.long)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. 提取空间权重
    print(f"\n--- 提取 {args.model} 的空间权重并映射到23个电极 ---")
    weights = get_spatial_weights_for_topomap(model, args.model, test_loader)
    
    print(f"通道权重统计:")
    print(f"  最大值: {np.max(weights):.4f}")
    print(f"  最小值: {np.min(weights):.4f}")
    print(f"  平均值: {np.mean(weights):.4f}")
    print(f"  标准差: {np.std(weights):.4f}")
    
    # 5. 创建电极位置信息
    try:
        info = mne.create_info(ch_names=CH_NAMES_23, sfreq=256, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
    except Exception as e:
        print(f"创建电极信息失败: {e}")
        return
    
    # 6. 绘制拓扑图
    print("\n--- 绘制拓扑图 ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    abs_weights = np.abs(weights)
    vmax = max(abs_weights.max(), 0.1)
    
    try:
        # 左图: 原始权重 (有正负)
        im1 = mne.viz.plot_topomap(
            data=weights, 
            pos=info, 
            names=info.ch_names, 
            cmap='RdBu_r', 
            sensors=True, 
            axes=ax1, 
            vlim=(-vmax, vmax),
            show=False,
            ch_type='eeg',
            size=2.5,
            contours=0
        )
        ax1.set_title(f"{args.model} Spatial Feature Weights\n(Red=Positive, Blue=Negative)", fontsize=32, pad=40)
        
        # 右图: 绝对值权重 (重要性)
        im2 = mne.viz.plot_topomap(
            data=abs_weights, 
            pos=info, 
            names=info.ch_names, 
            cmap='Reds', 
            sensors=True, 
            axes=ax2, 
            vlim=(0, vmax),
            show=False,
            ch_type='eeg',
            size=2.5,
            contours=0
        )
        ax2.set_title("Channel Importance Distribution\n(Darker=More Important)", fontsize=32, pad=40)
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1[0], ax=ax1, shrink=0.8, label='Weight Value')
        cbar2 = plt.colorbar(im2[0], ax=ax2, shrink=0.8, label='Importance')
        
        # 放大颜色条标签
        cbar1.ax.tick_params(labelsize=28)
        cbar2.ax.tick_params(labelsize=28)
        cbar1.ax.set_ylabel('Weight Value', fontsize=30)
        cbar2.ax.set_ylabel('Importance', fontsize=30)
        
    except Exception as e:
        print(f"绘制拓扑图失败: {e}")
        return
    
    # 7. 保存和显示
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.85)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f'{args.output_dir}/{args.model}_topomap_analysis.png'
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 拓扑图已保存: {output_path}")
        
        # 显示最重要的通道
        top_indices = np.argsort(abs_weights)[-5:][::-1]
        print(f"\n📊 {args.model} 最重要的5个通道:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. {CH_NAMES_23[idx]}: {weights[idx]:.4f}")
        
        plt.show()
        
    except Exception as e:
        print(f"保存图片失败: {e}")

if __name__ == "__main__":
    main_visualize()