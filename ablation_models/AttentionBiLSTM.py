"""
多尺度注意力BiLSTM模型 - 适配CHB-MIT癫痫检测项目
整合自EEG-Epilepsy-Prediction项目的先进架构

主要特性:
- 通道注意力: 自适应选择重要的EEG频带特征
- 时间注意力: 多头自注意力机制，专注癫痫发作关键时刻  
- 双向LSTM: 提取时序上下文，捕捉前后时序依赖
- 内存优化: 支持8GB GPU运行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """通道注意力机制 - 用于EEG频带特征选择"""
    
    def __init__(self, in_features: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, T, F]
        b, t, f = x.size()
        
        # 在时间维度上进行全局池化
        avg_out = self.fc(self.avg_pool(x.transpose(1, 2)).squeeze(-1))  # [B, F]
        max_out = self.fc(self.max_pool(x.transpose(1, 2)).squeeze(-1))  # [B, F]
        
        attention = self.sigmoid(avg_out + max_out).unsqueeze(1)  # [B, 1, F]
        return x * attention, attention.squeeze(1)


class TemporalAttention(nn.Module):
    """时间注意力机制 - 用于癫痫发作模式识别"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        # x: [B, T, H]
        batch_size, seq_len, _ = x.size()
        
        # 多头注意力
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T] 用于广播
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores.masked_fill_(mask_expanded == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_linear(context)
        output = self.layer_norm(output + x)  # 残差连接
        
        return output, attention_weights.mean(dim=1)  # 跨头平均


class AttentionBiLSTM(nn.Module):
    """
    多尺度注意力BiLSTM - 适配CHB-MIT数据格式
    
    输入格式: [B, C, T] (批次, 通道, 时间点) - 与现有EEGNet兼容
    输出格式: [B, num_classes] - 与现有训练脚本兼容
    """
    
    def __init__(self, nb_classes=2, Chans=23, Samples=512,
                 hidden_dim=128, num_layers=2, dropout=0.15,
                 use_attention=True, attention_heads=4):
        """
        参数:
        - nb_classes: 分类数量 (默认2: 正常/癫痫)
        - Chans: EEG通道数 (CHB-MIT为23)
        - Samples: 时间点数 (如2秒*256Hz=512)
        - hidden_dim: LSTM隐藏层维度 (内存优化: 128)
        - num_layers: LSTM层数 (内存优化: 2)
        - dropout: Dropout比例
        - use_attention: 是否使用注意力机制
        - attention_heads: 注意力头数 (内存优化: 4)
        """
        super().__init__()
        
        self.Chans = Chans
        self.Samples = Samples
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # 特征提取层 - 将原始EEG信号转换为特征序列
        # 使用1D卷积提取时序特征
        self.feature_extractor = nn.Sequential(
            # 第一层: 时间卷积
            nn.Conv1d(Chans, 64, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            
            # 第二层: 深度卷积
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            
            # 第三层: 特征压缩
            nn.Conv1d(128, hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )
        
        # 计算卷积后的序列长度
        # 经过三次卷积: stride=4,2,2 总共16倍下采样
        self.conv_output_len = Samples // 16
        
        # 通道注意力 - 用于特征选择
        if use_attention:
            self.channel_attention = ChannelAttention(hidden_dim, reduction=4)
        
        # BiLSTM主干网络
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        lstm_output_dim = hidden_dim * 2  # 双向LSTM
        
        # 时间注意力
        if use_attention:
            self.temporal_attention = TemporalAttention(lstm_output_dim, num_heads=attention_heads)
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, nb_classes)
        )
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重以提高数值稳定性"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        前向传播
        
        输入: x [B, C, T] - 与EEGNet兼容的格式
        输出: [B, num_classes] - 分类logits
        """
        # x: [B, C, T] -> [B, C, T] 
        batch_size = x.size(0)
        
        # 特征提取: [B, C, T] -> [B, hidden_dim, T']
        features = self.feature_extractor(x)  # [B, hidden_dim, conv_output_len]
        
        # 转置为LSTM格式: [B, hidden_dim, T'] -> [B, T', hidden_dim]
        features = features.transpose(1, 2)  # [B, conv_output_len, hidden_dim]
        
        # 通道注意力 (可选)
        attention_info = {}
        if self.use_attention:
            features, channel_weights = self.channel_attention(features)
            attention_info['channel_attention'] = channel_weights
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(features)  # [B, T', hidden_dim*2]
        
        # 时间注意力 (可选)
        if self.use_attention:
            attended_out, temporal_weights = self.temporal_attention(lstm_out)
            attention_info['temporal_attention'] = temporal_weights
            final_features = attended_out
        else:
            final_features = lstm_out
        
        # 全局池化: [B, T', hidden_dim*2] -> [B, hidden_dim*2]
        # 转置后池化
        pooled = self.global_pool(final_features.transpose(1, 2)).squeeze(-1)
        
        # 分类
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def get_attention_weights(self, x):
        """
        获取注意力权重用于可视化
        
        返回: 包含通道注意力和时间注意力权重的字典
        """
        if not self.use_attention:
            return None
            
        batch_size = x.size(0)
        
        # 特征提取
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)
        
        # 通道注意力
        features, channel_weights = self.channel_attention(features)
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(features)
        
        # 时间注意力
        attended_out, temporal_weights = self.temporal_attention(lstm_out)
        
        return {
            'channel_attention': channel_weights.detach().cpu().numpy(),
            'temporal_attention': temporal_weights.detach().cpu().numpy()
        }
    
    def get_model_size(self):
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }


# 为了向后兼容，创建一个别名
MultiScaleAttentionBiLSTM = AttentionBiLSTM


if __name__ == "__main__":
    # 测试模型
    model = AttentionBiLSTM(
        nb_classes=2,
        Chans=23,
        Samples=512,
        hidden_dim=128,
        num_layers=2,
        use_attention=True,
        attention_heads=4
    )
    
    # 打印模型信息
    print("模型结构:")
    print(model)
    
    print("\n模型参数:")
    size_info = model.get_model_size()
    print(f"总参数数: {size_info['total_params']:,}")
    print(f"可训练参数: {size_info['trainable_params']:,}")
    print(f"模型大小: {size_info['size_mb']:.2f} MB")
    
    # 测试前向传播
    x = torch.randn(2, 23, 512)  # [batch_size, channels, samples]
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        
        # 测试注意力权重
        attention_weights = model.get_attention_weights(x)
        if attention_weights:
            print(f"通道注意力权重形状: {attention_weights['channel_attention'].shape}")
            print(f"时间注意力权重形状: {attention_weights['temporal_attention'].shape}")