import torch
import random
import numpy as np
import torch.nn as nn
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
    return data_loader
