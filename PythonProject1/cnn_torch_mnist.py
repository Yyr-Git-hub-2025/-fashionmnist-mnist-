import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

class LeNet5(nn.Module):
    """
    LeNet-5 经典卷积神经网络结构，适用于28x28灰度图片（如MNIST）。
    主要包含两层卷积（Conv2d）+池化（MaxPool2d）和三层全连接（Linear）。
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积：输入 1通道，输出6通道，5x5卷积核
        # 卷积原理：卷积核在输入图像上滑动，每次计算一个局部加权和，从而提取局部特征。
        self.conv1 = nn.Conv2d(1, 6, 5)    # 输入1x28x28，输出6x24x24

        # 池化层（最大池化2x2，步长2）：对每个2x2区域取最大值，实现空间下采样和抗干扰
        self.pool = nn.MaxPool2d(2, 2)     # 6x24x24 -> 6x12x12

        # 第二层卷积：输入6通道，输出16通道，5x5卷积核
        # 提取更深层次、更复杂的特征
        self.conv2 = nn.Conv2d(6, 16, 5)   # 6x12x12 -> 16x8x8

        # 第二次池化：16x8x8 -> 16x4x4
        # 三个全连接层，最后输出10类
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积+激活+池化，提取空间局部特征并降维
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积和池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积和池化
        # 展平为一维向量以供全连接层使用
        x = x.view(-1, 16*4*4)
        # 全连接层+激活
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层（未激活，交给损失函数做softmax）
        x = self.fc3(x)
        return x

def main():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet5().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accs = []

    start = time.time()  # 计时开始

    for epoch in range(20):
        epoch_start = time.time()  # 每个epoch计时
        net.train()
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            out = net(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i}: Loss={loss.item():.4f}")
                losses.append(loss.item())
        # 测试准确率
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                out = net(img)
                pred = torch.argmax(out, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        acc = correct / total
        accs.append(acc)
        epoch_end = time.time()
        print(f"Epoch {epoch} Test Accuracy: {acc:.4f}")
        print(f"Epoch {epoch} Time: {epoch_end - epoch_start:.2f} seconds")  # 打印每个epoch耗时

    end = time.time()
    print(f"Total training time: {end - start:.2f} seconds")  # 打印总耗时

    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.show()

if __name__ == "__main__":
    main()