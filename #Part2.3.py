import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Hyper-parameters
num_epochs = 50  # 增加epochs以获得更好收敛
learning_rate = 1e-4  # 更小的学习率
batch_size = 128
channels = 3
depth = 128  # 增加卷积层深度
num_classes = 10
weight_decay = 1e-4  # 添加权重衰减（L2正则化）

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义更深的卷积网络，使用更大深度和更丰富的层
class ConvNetwork(nn.Module):
    def __init__(self, dim, in_channels, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(dim*4, dim*8, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(dim*8, dim*16, kernel_size=3, padding=1)  # 新增卷积层

        self.batch_norm1 = nn.BatchNorm2d(dim)
        self.batch_norm2 = nn.BatchNorm2d(dim*2)
        self.batch_norm3 = nn.BatchNorm2d(dim*4)
        self.batch_norm4 = nn.BatchNorm2d(dim*8)
        self.batch_norm5 = nn.BatchNorm2d(dim*16)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(dim*16*1*1, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.batch_norm1(nn.ReLU()(self.conv1(x))))
        x = self.pool(self.batch_norm2(nn.ReLU()(self.conv2(x))))
        x = self.pool(self.batch_norm3(nn.ReLU()(self.conv3(x))))
        x = self.pool(self.batch_norm4(nn.ReLU()(self.conv4(x))))
        x = self.pool(self.batch_norm5(nn.ReLU()(self.conv5(x))))  # 通过新增的卷积层

        x = x.view(-1, 128 * 16 * 1 * 1)  # Flatten
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)
        return x

# 检查MPS或GPU的可用性
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 实例化模型并转移到适当设备
model = ConvNetwork(dim=depth, in_channels=channels, num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# 训练模型
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    scheduler.step(epoch_loss)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')