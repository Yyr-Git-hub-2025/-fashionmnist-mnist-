import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=128, dataset='MNIST'):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'MNIST':
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def plot_samples(dataset='MNIST'):
    if dataset == 'MNIST':
        ds = datasets.MNIST(root='./data', train=True, download=True)
        classes = ds.classes
    else:
        ds = datasets.FashionMNIST(root='./data', train=True, download=True)
        classes = ds.classes
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        img, label = ds[i]
        axes[i].imshow(np.array(img), cmap='gray')
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    plot_samples('MNIST')
    plot_samples('FashionMNIST')