import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Manager
# from torch.utils.data import DataLoader, TensorDataset

from models import build_model
from dataloaders import dummy_dataloader

# Set seed for reproducibility
seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # Generate data
# def generate_data(num_samples):
#     x = np.linspace(-1, 1, num_samples)
#     y = x ** 3
#     return torch.tensor(x, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Divide a tensor into a specific number of sub-batches
def divide_into_sub_batches(tensor, num_sub_batches):
    sub_batch_size = tensor.size(0) // num_sub_batches
    return [tensor[i:i + sub_batch_size] for i in range(0, tensor.size(0), sub_batch_size)]

# Worker process
def worker(model, data, target, gradients_list, loss_list, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    gradients_list.append(gradients)
    loss_list.append(loss.item())

if __name__ == '__main__':
    manager = Manager()

    n_epochs = 5  # Number of epochs
    num_sub_batches = 4  # Number of smaller data batches, also equal to the number of worker

    # Create DataLoader
    num_samples = 200
    data_loader = dummy_dataloader(f=lambda x: x ** 3, num_samples=num_samples)

    # Initialize global model and optimizer
    global_model = build_model(arch="simplemodel", class_number=1)
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)


    for epoch in range(n_epochs):
        epoch_loss_list = []
        for iteration, (data_batch, target_batch) in enumerate(data_loader):
            gradients_list = manager.list()
            loss_list = manager.list()

            # Divide the data_batch and target_batch into smaller batches // again simulating DDP
            data_sub_batches = divide_into_sub_batches(data_batch, num_sub_batches)
            target_sub_batches = divide_into_sub_batches(target_batch, num_sub_batches)

            processes = []
            for data_sub_batch, target_sub_batch in zip(data_sub_batches, target_sub_batches):
                p = Process(target=worker, args=(global_model, data_sub_batch, target_sub_batch, gradients_list, loss_list, optimizer))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Aggregate gradients
            aggregated_gradients = [sum(grad) for grad in zip(*gradients_list)]

            # Update global model
            optimizer.zero_grad()
            for p, agg_grad in zip(global_model.parameters(), aggregated_gradients):
                p.grad = agg_grad
            optimizer.step()

            avg_iter_loss = sum(loss_list) / len(loss_list)
            epoch_loss_list.append(avg_iter_loss)
            print(f'Epoch: {epoch+1}, sub-batch: {iteration}, avg sub-batch loss: {round(avg_iter_loss, 3)}')

        # Compute and print the average epoch loss
        avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        print(f'Epoch: {epoch+1} done, avg loss: {round(avg_epoch_loss, 3)}')
