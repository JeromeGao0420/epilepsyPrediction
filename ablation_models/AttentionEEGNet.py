import torch
import torch.nn as nn

# --- 1. Squeeze-and-Excitation (SE) 通道注意力模块 ---
# 这个模块负责计算每个特征通道的重要性，用于定位的可解释性。
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        """
        channel: 输入特征通道数 (这里是 F1 * D)
        reduction: 降维比率，减少参数量
        """
        super(SEBlock, self).__init__()
        # 1. Squeeze (挤压): 全局平均池化，将 (B, C, H, W) 变为 (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Excitation (激励): 两个全连接层计算权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 权重在 0 到 1 之间
        )
        # 用于临时存储注意力权重，方便后续在 forward 中返回
        self.attention_weights = None 

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 挤压操作：从空间维度聚合信息
        y = self.avg_pool(x).view(b, c) 
        
        # 激励操作：计算每个通道的权重
        y = self.fc(y).view(b, c, 1, 1)
        
        # 存储权重，用于可视化。我们将 y 压缩成 (B, C) 的形状以便处理
        self.attention_weights = y.squeeze(-1).squeeze(-1)

        # 乘回原输入，加强重要的通道，削弱不重要的通道
        return x * y.expand_as(x)


# --- 2. 带有注意力机制的 EEGNet 模型 ---
class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=23, Samples=512, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        
        super(EEGNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.D = D
        self.F1 = F1

        # Block 1: Temporal Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        # Block 2: Depthwise Convolution (Spatial Filtering)
        self.conv2_spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
        )
        
        # 💡 创新点：插入通道注意力模块 💡
        # Attention channel count is F1 * D
        self.attention_module = SEBlock(channel=F1 * D, reduction=8)
        
        # Block 2 remainder
        self.conv2_pool_drop = nn.Sequential(
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 3: Separable Convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        # Classifier
        linear_in = F2 * (self.Samples // 32)
        self.linear = nn.Linear(linear_in, nb_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            self.linear
        )

    def forward(self, x):
        # 1. Block 1
        x = self.conv1(x)
        
        # 2. Block 2 Spatial Filtering
        x = self.conv2_spatial(x)
        
        # 3. 💡 Attention Module 💡
        # x_attended is the feature map enhanced by attention
        x_attended = self.attention_module(x)
        
        # 4. Block 2 Remainder
        x = self.conv2_pool_drop(x_attended)
        
        # 5. Block 3
        x = self.conv3(x)
        
        # 6. Classifier
        x = self.classifier(x)
        
        # 7. 返回值修改：同时返回分类结果和注意力权重
        # attention_weights 形状是 (Batch, F1*D)
        attention_map = self.attention_module.attention_weights 
        
        return x, attention_map 

# --- 测试代码 ---  
if __name__ == "__main__":  
    # 假设我们有 Batch=32, 23个通道, 512个时间点  
    input_data = torch.randn(32, 1, 23, 512)  
    model = EEGNet(nb_classes=2, Chans=23, Samples=512)  
    output, attention_map = model(input_data)  
    
    # 预期形状: output [32, 2], attention_map [32, F1*D] (例如 [32, 16])
    print("模型输出形状:", output.shape) 
    print("注意力图形状:", attention_map.shape) 
    print("Attention-EEGNet 模型构建成功！")