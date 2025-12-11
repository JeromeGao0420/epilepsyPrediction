"""
TCFormer模型实现 - 简化版本适配癫痫检测
基于Altaheri et al. (2025) "Temporal Convolutional Transformer for EEG Based Motor Imagery Decoding"

主要特性:
- 多核卷积前端 (Multi-Kernel CNN)
- Transformer编码器
- 时间卷积网络头部 (TCN)
- 适配CHB-MIT癫痫检测数据格式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiKernelConvBlock(nn.Module):
    """
    多核卷积块 - 使用不同核大小提取多尺度时间特征
    """
    
    def __init__(self, n_channels, temp_kernels=(16, 32, 64), F1=16, D=2, 
                 pool_length=8, dropout=0.3):
        super().__init__()
        
        self.n_channels = n_channels
        self.temp_kernels = temp_kernels
        self.F1 = F1
        self.D = D
        
        # 多核时间卷积层
        self.temp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F1, (1, kernel_size), padding=(0, kernel_size//2), bias=False),
                nn.BatchNorm2d(F1)
            ) for kernel_size in temp_kernels
        ])
        
        # 空间卷积层（深度卷积）
        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1*D),
                nn.ELU(),
                nn.AvgPool2d((1, pool_length)),
                nn.Dropout(dropout)
            ) for _ in temp_kernels
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(F1*D*len(temp_kernels), F1*D, 1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU()
        )
    
    def forward(self, x):
        # x: [B, C, T] -> [B, 1, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 多核时间卷积
        temp_features = []
        for i, (temp_conv, spatial_conv) in enumerate(zip(self.temp_convs, self.spatial_convs)):
            # 时间卷积
            temp_out = temp_conv(x)  # [B, F1, C, T']
            
            # 空间卷积
            spatial_out = spatial_conv(temp_out)  # [B, F1*D, 1, T'']
            temp_features.append(spatial_out)
        
        # 特征拼接和融合
        concat_features = torch.cat(temp_features, dim=1)  # [B, F1*D*n_kernels, 1, T'']
        fused_features = self.feature_fusion(concat_features)  # [B, F1*D, 1, T'']
        
        return fused_features.squeeze(2)  # [B, F1*D, T'']


class MultiHeadAttention(nn.Module):
    """
    简化的多头注意力机制
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置以便计算注意力
        Q = Q.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        context = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len, head_dim]
        
        # 重新组织并输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.out_linear(context)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    """
    
    def __init__(self, d_model, num_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = d_model * 4
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 自注意力 + 残差连接
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class TCNBlock(nn.Module):
    """
    时间卷积网络块
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        # 第一层卷积
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 第二层卷积
        out = self.conv2(out)
        out = self.norm2(out)
        
        # 因果填充（移除未来信息）
        if self.conv2.padding[0] > 0:
            out = out[:, :, :-self.conv2.padding[0]]
        
        # 残差连接 - 确保尺寸匹配
        residual = x if self.residual is None else self.residual(x)
        
        # 调整残差尺寸以匹配主路径
        if residual.size(2) != out.size(2):
            if residual.size(2) > out.size(2):
                residual = residual[:, :, :out.size(2)]
            else:
                # 如果残差比主路径短，用零填充
                padding_size = out.size(2) - residual.size(2)
                residual = torch.nn.functional.pad(residual, (0, padding_size))
        
        # 残差连接
        out = out + residual
        out = self.activation(out)
        
        return out


class SimplifiedTCFormer(nn.Module):
    """
    简化版TCFormer - 适配CHB-MIT癫痫检测数据格式
    
    输入格式: [B, C, T] (批次, 通道, 时间点) - 与现有EEGNet兼容
    输出格式: [B, nb_classes] - 与现有训练脚本兼容
    """
    
    def __init__(self, nb_classes=2, Chans=23, Samples=512, 
                 temp_kernels=(16, 32, 64), F1=16, D=2,
                 d_model=64, num_heads=8, num_layers=4,
                 tcn_channels=32, tcn_layers=2,
                 dropout=0.3):
        """
        参数:
        - nb_classes: 分类数量 (默认2: 正常/癫痫)
        - Chans: EEG通道数 (CHB-MIT为23)
        - Samples: 时间点数 (如2秒*256Hz=512)
        - temp_kernels: 时间卷积核大小
        - F1: 第一层滤波器数量
        - D: 深度乘数
        - d_model: Transformer模型维度
        - num_heads: 注意力头数
        - num_layers: Transformer层数
        - tcn_channels: TCN通道数
        - tcn_layers: TCN层数
        - dropout: Dropout比例
        """
        super().__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.d_model = d_model
        
        # 1. 多核卷积前端
        self.conv_frontend = MultiKernelConvBlock(
            Chans, temp_kernels, F1, D, pool_length=8, dropout=dropout
        )
        
        # 计算卷积后的特征维度
        conv_out_channels = F1 * D
        
        # 2. 特征投影到Transformer维度
        self.feature_projection = nn.Linear(conv_out_channels, d_model)
        
        # 3. Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 4. TCN头部
        self.tcn_layers = nn.ModuleList()
        tcn_in_channels = d_model
        
        for i in range(tcn_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                TCNBlock(tcn_in_channels, tcn_channels, 
                        kernel_size=3, dilation=dilation, dropout=dropout)
            )
            tcn_in_channels = tcn_channels
        
        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_channels, tcn_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels // 2, nb_classes)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        输入: x [B, C, T] - (批次大小, 通道数, 时间点)
        输出: [B, nb_classes] - 分类logits
        """
        # 1. 多核卷积特征提取
        conv_features = self.conv_frontend(x)  # [B, F1*D, T']
        
        # 2. 转换为Transformer输入格式
        B, C, T = conv_features.shape
        conv_features = conv_features.transpose(1, 2)  # [B, T', F1*D]
        
        # 3. 特征投影
        transformer_input = self.feature_projection(conv_features)  # [B, T', d_model]
        
        # 4. Transformer编码
        transformer_output = transformer_input
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)  # [B, T', d_model]
        
        # 5. 转换为TCN输入格式
        tcn_input = transformer_output.transpose(1, 2)  # [B, d_model, T']
        
        # 6. TCN处理
        tcn_output = tcn_input
        for tcn_layer in self.tcn_layers:
            tcn_output = tcn_layer(tcn_output)  # [B, tcn_channels, T']
        
        # 7. 分类
        output = self.classifier(tcn_output)  # [B, nb_classes]
        
        return output
    
    def get_feature_maps(self, x):
        """
        获取中间特征图用于可视化
        
        返回: 包含各层特征图的字典
        """
        features = {}
        
        # 卷积特征
        conv_features = self.conv_frontend(x)
        features['conv_features'] = conv_features.clone()
        
        # Transformer特征
        B, C, T = conv_features.shape
        conv_features_t = conv_features.transpose(1, 2)
        transformer_input = self.feature_projection(conv_features_t)
        
        transformer_output = transformer_input
        for i, layer in enumerate(self.transformer_layers):
            transformer_output = layer(transformer_output)
            features[f'transformer_layer_{i}'] = transformer_output.clone()
        
        # TCN特征
        tcn_input = transformer_output.transpose(1, 2)
        tcn_output = tcn_input
        for i, tcn_layer in enumerate(self.tcn_layers):
            tcn_output = tcn_layer(tcn_output)
            features[f'tcn_layer_{i}'] = tcn_output.clone()
        
        return features


# 测试函数
def test_tcformer():
    """测试TCFormer模型"""
    print("测试SimplifiedTCFormer模型...")
    
    # 创建模型
    model = SimplifiedTCFormer(
        nb_classes=2, 
        Chans=23, 
        Samples=512,
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
    
    # 创建测试数据
    batch_size = 4
    test_input = torch.randn(batch_size, 23, 512)
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
        features = model.get_feature_maps(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("特征图形状:")
    for name, feature in features.items():
        print(f"  {name}: {feature.shape}")
    
    print("SimplifiedTCFormer模型测试完成!")
    return True


if __name__ == "__main__":
    test_tcformer()