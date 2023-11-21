'''
This script is used to compute the mean and std of the dataset
'''

import torch
import random
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def get_cifar100_stats():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataloader = DataLoader(torchvision.datasets.CIFAR100(root='../data/', train=True, download=True, transform=transform), 
                              batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    means = []
    stdevs = []
    for X, _ in train_dataloader:
        # Dimensions 0,2,3 are respectively the batch, height and width dimensions
        means.append(X.mean(dim=(0,2,3)))
        stdevs.append(X.std(dim=(0,2,3)))
    mean = torch.stack(means, dim=0).mean(dim=0)
    stdev = torch.stack(stdevs, dim=0).mean(dim=0)
    return mean, stdev, X.shape

def get_cifar10_stats():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataloader = DataLoader(torchvision.datasets.CIFAR10(root='../data/', train=True, download=True, transform=transform), 
                              batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    means = []
    stdevs = []
    for X, _ in train_dataloader:
        # Dimensions 0,2,3 are respectively the batch, height and width dimensions
        means.append(X.mean(dim=(0,2,3)))
        stdevs.append(X.std(dim=(0,2,3)))
    mean = torch.stack(means, dim=0).mean(dim=0)
    stdev = torch.stack(stdevs, dim=0).mean(dim=0)
    return mean, stdev

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute dataset mean and std")
    parser.add_argument("-d", "--dataset", metavar="", type=str, default="cifar100", help="Dataset to use")
    args = parser.parse_args()

    if args.dataset == "cifar100":
        mean, std, shape = get_cifar100_stats()
        print("Dataset :", args.dataset)
        print("Shape   :", shape)
        print("Mean    :", mean)
        print("Std     :", std)
    elif args.dataset == "cifar10":
        mean, std = get_cifar10_stats()
        print("Dataset :", args.dataset)
        print("Mean    :", mean)
        print("Std     :", std)
    else:
        print("Dataset not supported")
