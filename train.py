from model import EEGNet  
import torch  
import torch.nn as nn  
  

model = EEGNet(nb_classes=2, Chans=22, Samples=512).cuda()  
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
# 开始训练循环  
