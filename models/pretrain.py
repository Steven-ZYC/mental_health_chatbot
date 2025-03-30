import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import os

# 创建保存目录
os.makedirs('models', exist_ok=True)

# 增强的数据预处理
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),  # 明确转换为单通道
    transforms.RandomHorizontalFlip(), #随机反转
    transforms.RandomRotation(10), #随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.225])
])


# 加载数据集
train_dataset = datasets.ImageFolder(root='archive/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='archive/test', transform=test_transform)

# 保存类别标签映射
with open('models/class_labels.json', 'w') as f:
    json.dump(train_dataset.class_to_idx, f)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


# 修改模型结构
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base_model = models.resnet18(weights='DEFAULT')
        # 第一层卷积接受单通道输入
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 全连接层输出为类别数
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = EmotionResNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)


# 训练函数
def train_model(epochs):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        val_loss = val_loss / len(test_dataset)
        val_acc = 100 * correct / total
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print('New best model saved!')

    print(f'Training complete. Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    train_model(epochs=100)