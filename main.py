import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Process, Manager, Pool
from multiprocessing.pool import ThreadPool
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from dataloaders import *

# Set seed for reproducibility
seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Divide a tensor into a specific number of sub-batches
def divide_into_sub_batches(tensor, num_sub_batches):
    sub_batch_size = tensor.size(0) // num_sub_batches
    return [tensor[i:i + sub_batch_size] for i in range(0, tensor.size(0), sub_batch_size)]

# Worker process
def worker(model, data, target, gradients_list, loss_list, optimizer, criterion):
    optimizer.zero_grad()
    output = model(data)
    # loss = nn.MSELoss()(output, target)
    loss = criterion(output, target)
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    gradients_list.append(gradients)
    loss_list.append(loss.item())

def parallel_worker_train(args):
    # this is a helper function for parallelizing worker
    model, data, target, gradients_list, loss_list, optimizer, criterion = args
    worker(model, data, target, gradients_list, loss_list, optimizer, criterion)

def setup_train_job(args):
    # Initialize global model and optimizer, and set up datalaoder
    dataset = args.dataset
    global_batch_size = args.global_batch_size
    if dataset=="dummy":
        data_loader = dummy_dataloader(f=lambda x: x ** 3, num_samples=500)
        model = build_model(arch="simplemodel", class_number=1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
    elif dataset.lower()=="mnist":
        # TODO: test loader currently not used!!!
        data_loader, test_loader = mnist_dataloader(global_batch_size)
        model = build_model(arch="mlp", class_number=10)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Dataset not supported")
    
    return data_loader, model, optimizer, criterion
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDP simulation")
    parser.add_argument("-d", "--dataset", metavar="", type=str, default="dummy", help="Dataset to use")
    parser.add_argument("-e", "--epoch", metavar="", type=int, default=3, help="Number of epochs")
    parser.add_argument("-gb", "--global_batch_size", metavar="", type=int, default=512, help="Global batch size")
    parser.add_argument("-w", "--worker", metavar="", type=int, default=4, 
                        help="Number of workers/sub-batches (Note: global batch size must be divisible by number of workers))")
    parser.add_argument("--proc", metavar="", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

    # parse arguments
    n_epochs = args.epoch
    num_sub_batches = args.worker
    num_processes = args.proc

    # print numbers worker
    print(f'Number of workers: {num_sub_batches}')

    # multiprocessing manager, and tensorboard writer
    manager = Manager()
    writer = SummaryWriter()

    # setup train job
    data_loader, global_model, optimizer, criterion = setup_train_job(args)

    training_iter = 0
    for epoch in range(n_epochs):
        epoch_loss_list = []
        for iteration, (data_batch, target_batch) in enumerate(data_loader):
            training_iter+=1
            optimizer.zero_grad()
            gradients_list = manager.list()
            loss_list = manager.list()

            # Divide the data_batch and target_batch into smaller batches // again simulating DDP
            data_sub_batches = divide_into_sub_batches(data_batch, num_sub_batches)
            target_sub_batches = divide_into_sub_batches(target_batch, num_sub_batches)
            # print(len(data_sub_batches))
            
            args_list = [(global_model, data, target, gradients_list, loss_list, optimizer, criterion) for data, target in zip(data_sub_batches, target_sub_batches)]
            with Pool(processes=num_processes) as pool:
                pool.map(parallel_worker_train, args_list)

            # Aggregate gradients
            aggregated_gradients = [sum(grad) for grad in zip(*gradients_list)]

            # Update global model
            for p, agg_grad in zip(global_model.parameters(), aggregated_gradients):
                p.grad = agg_grad
            optimizer.step()

            avg_iter_loss = sum(loss_list) / len(loss_list)
            epoch_loss_list.append(avg_iter_loss)
            # print(f'Epoch: {epoch+1}, sub-batch: {iteration}, avg sub-batch loss: {round(avg_iter_loss, 3)}')
            # print curr iter / total iter
            print(f'Epoch: {epoch+1}, sub-batch: {iteration+1}/{len(data_loader)}, avg sub-batch loss: {round(avg_iter_loss, 3)}')

            # Write to tensorboard tag with worker number
            # writer.add_scalar('Loss/worker', avg_iter_loss, iteration)
            writer.add_scalar('Loss/iter', avg_iter_loss, training_iter)

        # Compute and print the average epoch loss
        avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        # writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        print(f'Epoch: {epoch+1} done, avg loss: {round(avg_epoch_loss, 3)}')

    writer.close()