import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    """
    DeepConvNet 实现 (基于 Schirrmeister et al., 2017)

    说明:
    - 输入形状: (batch, 1, Chans, Samples)
    - 输出: logits (batch, nb_classes)
    - 这份实现用于消融实验，保持结构简单易读，可按需调整滤波器数量和池化策略。
    """
    def __init__(self, nb_classes, Chans=23, Samples=512, dropoutRate=0.5, kernLength=5):
        super(DeepConvNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples

        # Block 1: Temporal conv + Spatial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.Conv2d(25, 25, (Chans, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropoutRate)
        )

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropoutRate)
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropoutRate)
        )

        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropoutRate)
        )

        # 计算全连接输入维度: 池化四次 (每次时间轴 /2)，高度已为 1
        linear_in = 200 * (self.Samples // 16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_in, nb_classes)
        )

    def forward(self, x):
        """Forward pass

        Args:
            x: Tensor shape (B, 1, Chans, Samples)

        Returns:
            logits: Tensor shape (B, nb_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick smoke test
    BATCH = 8
    CHANS = 23
    SAMPLES = 512
    model = DeepConvNet(nb_classes=2, Chans=CHANS, Samples=SAMPLES)
    sample = torch.randn(BATCH, 1, CHANS, SAMPLES)
    out = model(sample)
    print("DeepConvNet output shape:", out.shape)  # expect [BATCH, 2]
