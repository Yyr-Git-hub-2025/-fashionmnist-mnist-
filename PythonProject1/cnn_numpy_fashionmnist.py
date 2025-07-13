import numpy as np
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 工具函数
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, label):
    N = pred.shape[0]
    log_likelihood = -np.log(pred[np.arange(N), label] + 1e-9)
    return np.sum(log_likelihood) / N

def accuracy(pred, label):
    return np.mean(np.argmax(pred, axis=1) == label)

# im2col和col2im工具
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入的4D数据(batch, channel, height, width)展开为2D矩阵，用于矢量化卷积/池化运算。
    每一行对应输入的一个滑动窗口区域。
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    将im2col展开的2D矩阵还原为原始4D输入格式。
    主要用于卷积/池化的反向传播阶段。
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H+pad, pad:W+pad]

# 矢量化Conv2D
class Conv2D:
    """
    二维卷积层。
    原理：通过im2col将输入和卷积核展开为2D矩阵，利用矩阵乘法实现所有滑窗区域的卷积计算，显著提升速度。
    卷积本质是用卷积核在输入特征图上滑动，对每个窗口区域做加权求和加偏置，生成输出特征图。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
            2. / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros((out_channels, 1))

    def forward(self, x):
        """
        前向传播：输入x [N, C, H, W]，输出特征图 [N, F, OH, OW]
        通过im2col将输入摊平，和卷积核权重矩阵做矩阵乘法，得到所有滑动窗口的卷积结果。
        """
        self.x = x
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape
        SH, SW = self.stride, self.stride
        PH, PW = self.padding, self.padding
        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

        self.col = im2col(x, KH, KW, SH, PH)                   # [N*OH*OW, C*KH*KW]
        self.col_W = self.W.reshape(F, -1).T                   # [C*KH*KW, F]
        out = np.dot(self.col, self.col_W) + self.b.ravel()    # [N*OH*OW, F]
        out = out.reshape(N, OH, OW, F).transpose(0,3,1,2)     # [N, F, OH, OW]
        return out

    def backward(self, dout, lr):
        """
        反向传播：计算输入梯度和权重、偏置的梯度，并进行参数更新。
        """
        N, C, H, W = self.x.shape
        F, _, KH, KW = self.W.shape
        SH, SW = self.stride, self.stride
        PH, PW = self.padding, self.padding
        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

        dout_ = dout.transpose(0,2,3,1).reshape(-1, F)
        dW = np.dot(self.col.T, dout_)
        dW = dW.T.reshape(F, C, KH, KW)
        db = np.sum(dout_, axis=0).reshape(F, 1)

        dcol = np.dot(dout_, self.col_W.T)
        dx = col2im(dcol, self.x.shape, KH, KW, SH, PH)

        self.W -= lr * dW / N
        self.b -= lr * db / N
        return dx

# 矢量化MaxPool2D
class MaxPool2D:
    """
    最大池化层。
    原理：用im2col收集每个池化窗口的像素，直接用np.max取最大值，得到下采样输出。
    反向传播时只将上游梯度传给正向最大值所在位置。
    """
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        前向：每个窗口区域取最大值，实现空间下采样。
        """
        self.x = x
        N, C, H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride
        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        col = im2col(x, KH, KW, SH, 0).reshape(N*OH*OW*C, KH*KW)
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, OH, OW, C).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        """
        反向：只将梯度传给正向最大值所在位置，其他位置为0。
        """
        N, C, H, W = self.x.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride
        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        dmax = np.zeros((N*OH*OW*C, KH*KW))
        dout_flat = dout.transpose(0,2,3,1).flatten()
        dmax[np.arange(dmax.shape[0]), self.arg_max] = dout_flat
        dmax = dmax.reshape(N, OH, OW, C, KH, KW).transpose(0,3,4,5,1,2)
        dx = np.zeros((N, C, H, W))
        for y in range(KH):
            y_max = y + SH*OH
            for x in range(KW):
                x_max = x + SW*OW
                dx[:, :, y:y_max:SH, x:x_max:SW] += dmax[:, :, y, x, :, :]
        return dx

class Flatten:
    """
    展平层，将多维特征数据拉平成2D以便接入全连接层。
    """
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.x_shape)

class Dense:
    """
    全连接层，对输入特征做线性变换：y = xW + b
    """
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros((1, out_dim))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout, lr):
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)
        self.W -= lr * dW / self.x.shape[0]
        self.b -= lr * db / self.x.shape[0]
        return dx

#  网络定义
class SimpleLeNet:
    """
    LeNet-5结构的简化实现，适用于28x28单通道图像（如MNIST/FashionMNIST）。
    """
    def __init__(self):
        self.conv1 = Conv2D(1, 6, 5)
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(6, 16, 5)
        self.pool2 = MaxPool2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(16 * 4 * 4, 120)
        self.fc2 = Dense(120, 84)
        self.fc3 = Dense(84, 10)

    def forward(self, x):
        self.conv1_out = self.conv1.forward(x)
        self.relu1_out = relu(self.conv1_out)
        self.pool1_out = self.pool1.forward(self.relu1_out)
        self.conv2_out = self.conv2.forward(self.pool1_out)
        self.relu2_out = relu(self.conv2_out)
        self.pool2_out = self.pool2.forward(self.relu2_out)
        self.flatten_out = self.flatten.forward(self.pool2_out)
        self.fc1_out = self.fc1.forward(self.flatten_out)
        self.relu3_out = relu(self.fc1_out)
        self.fc2_out = self.fc2.forward(self.relu3_out)
        self.relu4_out = relu(self.fc2_out)
        self.fc3_out = self.fc3.forward(self.relu4_out)
        return self.fc3_out

    def backward(self, dout, lr):
        dout = self.fc3.backward(dout, lr)
        dout = dout * relu_grad(self.fc2_out)
        dout = self.fc2.backward(dout, lr)
        dout = dout * relu_grad(self.fc1_out)
        dout = self.fc1.backward(dout, lr)
        dout = self.flatten.backward(dout)
        dout = self.pool2.backward(dout)
        dout = dout * relu_grad(self.conv2_out)
        dout = self.conv2.backward(dout, lr)
        dout = self.pool1.backward(dout)
        dout = dout * relu_grad(self.conv1_out)
        dout = self.conv1.backward(dout, lr)
        return dout

#  数据准备
def get_loader(batch_size=64, dataset='FASHIONMNIST'):
    if dataset.upper() == 'MNIST':
        ds = datasets.MNIST
    elif dataset.upper() == 'FASHIONMNIST':
        ds = datasets.FashionMNIST
    else:
        raise ValueError('Unsupported dataset')
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = ds(root='./data', train=True, download=False, transform=transform)
    test_set = ds(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def _wrapper(loader):
        for img, label in loader:
            yield img.numpy(), label.numpy()

    return list(_wrapper(train_loader)), list(_wrapper(test_loader))

#  训练与测试
def train(model, train_loader, test_loader, epochs=5, lr=0.01):
    losses, accs = [], []
    for epoch in range(epochs):
        epoch_start = time.time()
        for i, (img, label) in enumerate(train_loader):
            out = model.forward(img)
            pred = softmax(out)
            loss = cross_entropy_loss(pred, label)
            dout = pred.copy()
            dout[np.arange(len(label)), label] -= 1
            dout /= len(label)
            model.backward(dout, lr)
            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i}: Loss={loss:.4f}")
                losses.append(loss)
        acc = test(model, test_loader)
        accs.append(acc)
        epoch_end = time.time()
        print(f"Epoch {epoch} Test Accuracy: {acc:.4f}")
        print(f"Epoch {epoch} Time: {epoch_end - epoch_start:.2f} seconds")
    return losses, accs

def test(model, test_loader):
    all_pred, all_label = [], []
    for img, label in test_loader:
        out = model.forward(img)
        pred = np.argmax(softmax(out), axis=1)
        all_pred.append(pred)
        all_label.append(label)
    all_pred = np.concatenate(all_pred)
    all_label = np.concatenate(all_label)
    return np.mean(all_pred == all_label)

# 主程序
if __name__ == "__main__":
    train_loader, test_loader = get_loader(batch_size=64, dataset='FASHIONMNIST')
    model = SimpleLeNet()
    start = time.time()
    losses, accs = train(model, train_loader, test_loader, epochs=20, lr=0.01)
    end = time.time()
    print(f"Total training time: {end - start:.2f} seconds")
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show()