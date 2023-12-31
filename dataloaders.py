import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def dummy_dataloader(f, num_samples):
    x = np.linspace(-1, 1, num_samples)
    # y = x ** 3
    y = f(x)

    # x.shape: [num_samples, 1]
    # y.shape: [num_samples, 1]
    x_data, y_data = torch.tensor(x, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=40, shuffle=True)
    # create test data
    test_num_samples = 1000
    x_test = np.linspace(-1, 1, test_num_samples)
    y_test = f(x_test)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True)
    return data_loader, test_loader

def mnist_dataloader(global_batch_size, test_batch_size=256):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_loader = DataLoader(torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=trans), 
                              batch_size=global_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=trans), 
                              batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def CIFAR100_dataloader(global_batch_size, test_batch_size=256, arch=None):
    # set up CIFAR100 similar to MNIST
    if arch is None:
        transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5070, 0.4865, 0.4408), (0.2672, 0.2563, 0.2760))])
    elif "resnet" in arch.lower():
        transform = transforms.Compose(
                                    [transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5070, 0.4865, 0.4408), (0.2672, 0.2563, 0.2760))])
    train_loader = DataLoader(torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transform), 
                              batch_size=global_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transform), 
                              batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def CIFAR10_dataloader(global_batch_size, test_batch_size=256, arch=None):
    # set up CIFAR10 similar to MNIST
    MEAN = (0.4914, 0.4822, 0.4466)
    STD=  (0.2470, 0.2434, 0.2615)
    if arch is None:
        transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])
    elif "resnet" in arch.lower():
        transform = transforms.Compose(
                                    [transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    train_loader = DataLoader(torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform), 
                              batch_size=global_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transform), 
                              batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader