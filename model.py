import torch  
import torch.nn as nn  
  
class EEGNet(nn.Module):  
    def __init__(self, nb_classes, Chans=22, Samples=512,   
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):  
        """  
        参数说明:  
        nb_classes: 分类数量 (癫痫检测通常是2: 发作/不发作)  
        Chans: EEG通道数 (CHB-MIT数据集通常是22或23，根据你处理的数据定)  
        Samples: 时间点数 (如果你切片是2秒，采样率256Hz，这里就是512)  
        """  
        super(EEGNet, self).__init__()  
        self.Chans = Chans  
        self.Samples = Samples  
  
        # Block 1: 时间卷积 (Temporal Convolution)  
        self.conv1 = nn.Sequential(  
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),  
            nn.BatchNorm2d(F1)  
        )  
  
        # Block 2: 深度卷积 (Depthwise Convolution) - 这里做空间滤波，提取不同通道的关系  
        self.conv2 = nn.Sequential(  
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),  
            nn.BatchNorm2d(F1 * D),  
            nn.ELU(),  
            nn.AvgPool2d((1, 4)),  
            nn.Dropout(dropoutRate)  
        )  
        
        # 创新点这里加入自注意力机制
        
        # Block 3: 可分离卷积 (Separable Convolution)  
        self.conv3 = nn.Sequential(  
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  
            nn.BatchNorm2d(F2),  
            nn.ELU(),  
            nn.AvgPool2d((1, 8)),  
            nn.Dropout(dropoutRate)  
        )  
  
        # 全连接层分类  
        self.classifier = nn.Sequential(  
            nn.Flatten(),  
            nn.Linear(F2 * (Samples // 32), nb_classes) # 注意：Samples//32 取决于上面的两次池化(4*8=32)  
        )  
  
    def forward(self, x):  
        # 输入 x 的形状应该是 (Batch, 1, Chans, Samples)  
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = self.classifier(x)  
        return x  
  
# --- 测试代码 ---  
if __name__ == "__main__":  
    # 假设我们有 Batch=32, 22个通道, 512个时间点  
    input_data = torch.randn(32, 1, 22, 512)  
    model = EEGNet(nb_classes=2, Chans=22, Samples=512)  
    output = model(input_data)  
    print("模型输出形状:", output.shape) # 应该是 [32, 2]  
    print("Baseline 模型构建成功！")  
