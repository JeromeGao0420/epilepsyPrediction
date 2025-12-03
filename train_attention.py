import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
import time

# 导入您在 model.py 中定义的 EEGNet 模型
from ablation_models import AttentionEEGNet

# --- 1. 参数设置 ---
# 请根据您的实际数据修改以下参数
BATCH_SIZE = 64        # 批次大小
LEARNING_RATE = 1e-3   # 学习率
NUM_EPOCHS = 50        # 训练轮次
CHANS = 23             # 通道数 (根据您的 CHB-MIT 数据确定)
SAMPLES = 512          # 每个切片的时间点数 (例如 2秒 * 256Hz)
NUM_CLASSES = 2        # 分类数 (0:正常, 1:癫痫)
# 检测设备类型 - 优先使用MPS(Mac M1/M2/M3), 然后CUDA, 最后CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("-> 使用MPS设备进行训练 (Apple Silicon GPU)")
    print("-> 注意：MPS仅支持float32，所有数据将自动转换为float32")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("-> 使用CUDA设备进行训练")
else:
    DEVICE = torch.device("cpu")
    print("-> 使用CPU设备进行训练")

# --- 2. MPS兼容性函数 ---
def ensure_mps_compatibility(tensor, target_dtype=torch.float32):
    """
    确保tensor与MPS框架兼容，强制转换为指定的数据类型
    MPS不支持float64，需要转换为float32
    """
    if tensor.dtype == torch.float64:
        tensor = tensor.to(target_dtype)
    return tensor

def safe_to_device(tensor, device):
    """
    安全地将tensor移动到指定设备，确保MPS兼容性
    """
    if device.type == 'mps':
        tensor = ensure_mps_compatibility(tensor)
    return tensor.to(device)

# --- 3. 数据加载函数 (保持不变) ---
def load_eeg_data(path='./data/'):
    """
    加载预处理好的训练集和测试集数据。
    """
    print("-> 正在加载数据...")
    try:
        X_train = np.load(path + 'X_train.npy')
        Y_train = np.load(path + 'Y_train.npy')
        X_test = np.load(path + 'X_test.npy')
        Y_test = np.load(path + 'Y_test.npy')
    except FileNotFoundError:
        print("错误：数据文件未找到！请检查路径和文件名是否正确。")
        return None, None
    
    X_train = torch.from_numpy(X_train).to(torch.float32).unsqueeze(1)
    Y_train = torch.from_numpy(Y_train).to(torch.long)
    X_test = torch.from_numpy(X_test).to(torch.float32).unsqueeze(1)
    Y_test = torch.from_numpy(Y_test).to(torch.long)
    
    if DEVICE.type == 'mps':
        X_train = ensure_mps_compatibility(X_train)
        X_test = ensure_mps_compatibility(X_test)
        print("-> 已确保数据与MPS设备兼容 (float32)")

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print(f"数据形状: {X_train.shape}")
    print("-> 数据加载完成。")
    return train_loader, test_loader


# --- 4. 训练和验证函数 (关键修改) ---
def train_epoch(model, dataloader, criterion, optimizer):
    """单次训练循环"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for inputs, labels in dataloader:
        inputs = safe_to_device(inputs, DEVICE)
        labels = safe_to_device(labels, DEVICE)

        optimizer.zero_grad()
        # ⚠️ 修改：模型现在返回两个值
        outputs, _ = model(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(dataloader)
    avg_acc = correct_predictions.float() / len(dataloader.dataset)
    return avg_loss, avg_acc

def evaluate(model, dataloader, criterion):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = safe_to_device(inputs, DEVICE)
            labels = safe_to_device(labels, DEVICE)
            
            # ⚠️ 修改：模型现在返回两个值
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    acc = accuracy_score(all_labels, all_preds)
    sen = recall_score(all_labels, all_preds, pos_label=1) 
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    
    return avg_loss, acc, sen, f1


# --- 5. 主函数 (保持不变) ---
def main():
    # 1. 初始化模型、损失函数和优化器
    # ⚠️ 注意：这里我们使用 Chans=23
    model = AttentionEEGNet.EEGNet(nb_classes=NUM_CLASSES, Chans=CHANS, Samples=SAMPLES).to(DEVICE)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    # 2. 加载数据
    train_loader, test_loader = load_eeg_data()
    if train_loader is None:
        return

    print("\n--- 开始训练 ---")
    best_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # 验证
        val_loss, val_acc, val_sen, val_f1 = evaluate(model, test_loader, criterion)
        
        end_time = time.time()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {end_time - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Sen: {val_sen:.4f} | Val F1: {val_f1:.4f}")

        # 保存 F1-Score 最好的模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            # 将模型保存到文件，以便将来加载和使用
            torch.save(model.state_dict(), './best_eegnet_model_attention.pth')
            print(f"*** 模型性能提升, F1-Score: {best_f1:.4f}, 已保存! ***")

    print("\n--- 训练结束 ---")
    # 训练结束后可直接进入可视化或后续分析步骤，无需重新加载模型
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行发生错误: {e}")