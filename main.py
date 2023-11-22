import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from multiprocessing import Process, Manager, Pool
from multiprocessing.pool import ThreadPool
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from dataloaders import *
import aggregation_rules
import data_distributor

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
def worker(idx, model, data, target, gradients_dict, loss_list, optimizer, criterion, faulty):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    # loss = nn.MSELoss()(output, target)
    loss = criterion(output, target)
    loss.backward()
    if faulty:
        # TODO: current hard coded noise type!!!

        # print("yoyooyo")
        # determ
        # random_scaler = random.randint(2, 10)
        # random_constant = random.randint(2, 10)
        gradients = [ (p.grad.clone() * (10)) for p in model.parameters()]
        
        # add gaussian noise to gradients
        # gradients = [p.grad+(torch.randn_like(p.grad.clone())) for p in model.parameters()]

        # Random
        # gradients = [ torch.randn_like(p.grad.clone()) for p in model.parameters()]

        # gradients = [torch.ones_like(p.grad) * 100000 for p in model.parameters()]
        # convert gradients to numpy array
    else:
        gradients = [p.grad.clone() for p in model.parameters()]
    
    # gradients_list.append(gradients)
    gradients_dict[idx] = gradients
    loss_list.append(loss.item())

def parallel_worker_train(args):
    # this is a helper function for parallelizing worker
    idx, model, data, target, gradients_list, loss_list, optimizer, criterion, faulty = args
    worker(idx, model, data, target, gradients_list, loss_list, optimizer, criterion, faulty)

def setup_train_job(args):
    # Initialize global model and optimizer, and set up datalaoder
    dataset = args.dataset
    global_batch_size = args.global_batch_size
    if dataset=="dummy":
        # test_loader = None
        data_loader, test_loader = dummy_dataloader(f=lambda x: x ** 3, num_samples=500)
        if args.arch is not None:
            model = build_model(arch=args.arch, class_number=1, pretrained=True)
        else:
            model = build_model(arch="simplemodel", class_number=1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
    elif dataset.lower()=="mnist":
        data_loader, test_loader = mnist_dataloader(global_batch_size)
        if args.arch is not None:
            model = build_model(arch=args.arch, class_number=10, pretrained=True)
        else:
            model = build_model(arch="mlp", class_number=10)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    elif dataset.lower()=="cifar100":
        if args.arch is not None:
            data_loader, test_loader = CIFAR100_dataloader(global_batch_size, arch=args.arch)
            model = build_model(arch=args.arch, class_number=100, pretrained=True)
        else:
            data_loader, test_loader = CIFAR100_dataloader(global_batch_size)
            model = build_model(arch="SimpleCNN", class_number=100)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    elif dataset.lower()=="cifar10":
        if args.arch is not None:
            data_loader, test_loader = CIFAR10_dataloader(global_batch_size, arch=args.arch)
            model = build_model(arch=args.arch, class_number=10, pretrained=True)
        else:
            data_loader, test_loader = CIFAR10_dataloader(global_batch_size)
            model = build_model(arch="SimpleCNN", class_number=10)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Dataset not supported")
    
    return data_loader, test_loader, model, optimizer, criterion
    
def inference(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    for data_batch, target_batch in tqdm(data_loader, desc="validation"):
    # for data_batch, target_batch in data_loader:
        output = model(data_batch.to(device))
        _, predicted = torch.max(output.data, 1)
        # print("pred", predicted)
        # print("gt", target_batch)
        total += target_batch.size(0)
        correct += (predicted.detach().cpu() == target_batch).sum().item()
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDP simulation")
    parser.add_argument("-d", "--dataset", metavar="", type=str, default="dummy", help="Dataset to use")
    parser.add_argument("--arch", metavar="", type=str, default=None, help="Model to use")
    parser.add_argument("-e", "--epoch", metavar="", type=int, default=3, help="Number of epochs")
    parser.add_argument("-gb", "--global_batch_size", metavar="", type=int, default=512, help="Global batch size")
    parser.add_argument("-w", "--worker", metavar="", type=int, default=4, 
                        help="Number of workers/sub-batches (Note: global batch size must be divisible by number of workers))")
    # take in a list of faulty workers idx
    parser.add_argument("-f", "--faulty", metavar="", type=int, nargs='+', default=[], help="Indics of faulty worker (Ex: -f 0 1 2)")
    parser.add_argument("-df", "--defense", metavar="", type=str, default=None, help="Defense method")
    parser.add_argument("--proc", metavar="", type=int, default=1, help="Number of processes")
    parser.add_argument("-tb", "--tb", metavar="", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument("--device", metavar="", type=str, default="cpu", help="Device to use")
    # correct or not, use action
    parser.add_argument("--correct", action="store_true", help="Whether to use error correction")
    # data collection path
    parser.add_argument("--data_coll_path", metavar="", type=str, default=None, help="Path to data collection")
    args = parser.parse_args()

    # parse arguments
    n_epochs = args.epoch
    num_sub_batches = args.worker
    faulty_worker_idxs = args.faulty
    defense_method = args.defense
    num_processes = args.proc
    tb_log_dir = args.tb
    device = args.device
    error_correction = args.correct

    if args.data_coll_path is None:
        data_coll_path = False
    else:
        error_correction = True
        data_coll_path = args.data_coll_path

    # set device
    print(f'Device: {device}')
    if device == "cpu":
        device = torch.device("cpu")
    elif device == "gpu":
        device = torch.device("cuda:3")
        # raise ValueError("Sharing CUDA tensor across processes is not yet supported")
    elif device == "mps":
        # have bugs when using mps
        device = torch.device("mps")
    else:
        raise ValueError("Device not supported")

    # print numbers worker, and proc
    print(f'Number of workers: {num_sub_batches}')
    print(f'Number of processes: {num_processes}')
    print(f'Faulty worker idxs: {faulty_worker_idxs}')

    # print args
    print(args)
    print()
    
    # multiprocessing manager, and tensorboard writer
    manager = Manager()

    # Tensorboard writer
    if tb_log_dir is not None:
        writer = SummaryWriter(log_dir=tb_log_dir)

    # setup train job
    data_loader, test_loader, global_model, optimizer, criterion = setup_train_job(args)
    global_model.to(device)

    for param_group in optimizer.param_groups: base_lr = param_group['lr'] 

    # set up aggregator, and data distributor
    correct_args = {
        'correct': error_correction,
        'cmodel': "linear",
        'model': global_model,
        'data_coll_path': data_coll_path
    }
    aggregator = aggregation_rules.Aggregator(device=device, correct_args=correct_args)
    data_distributor = data_distributor.DDPDataDistributor(num_workers=num_sub_batches, 
                                                           faulty_worker_ids=faulty_worker_idxs, 
                                                           correction=error_correction)
    aggregator.update_faulty_worker_idxs(faulty_worker_idxs)

    corrected = False
    training_iter = 0
    for epoch in range(n_epochs):
        epoch_loss_list = []
        for iteration, (data_batch, target_batch) in enumerate(data_loader):
            training_iter+=1
            global_model.train()
            optimizer.zero_grad()

            # gradients_dict = manager.dict()
            # loss_list = manager.list()
            gradients_dict = {}
            loss_list = []

            # Divide the data_batch and target_batch into smaller batches // again simulating DDP
            # data_sub_batches = divide_into_sub_batches(data_batch, num_sub_batches)
            # target_sub_batches = divide_into_sub_batches(target_batch, num_sub_batches)
            data_sub_batches, target_sub_batches, worker_batch_map = data_distributor.distribute(data_batch, target_batch, corrected)
            print("W-B MAP:", worker_batch_map)
            
            faulty = False
            args_list = []
            for i, (data, target) in enumerate(zip(data_sub_batches, target_sub_batches)):
                data, target = data.to(device), target.to(device)
                if i in faulty_worker_idxs: faulty = True
                else: faulty = False
                args_list.append((i, global_model, data, target, gradients_dict, loss_list, optimizer, criterion, faulty))
            
            # Multi-processing
            # with Pool(processes=num_processes) as pool:
            #     pool.map(parallel_worker_train, args_list)
            # with ThreadPool(processes=num_processes) as pool:
            #     pool.map(parallel_worker_train, args_list)
            # single process
            for args in args_list:
                parallel_worker_train(args)

            # Aggregate gradients
            if defense_method == "krum":
                print("krum defense")
                gradients_list = aggregation_rules.krum(gradients_dict, num_sub_batches, f=1)
                print(len(gradients_list))
            elif defense_method == "mean":
                print("mean defense")
                gradients_list = aggregation_rules.mean_filter(gradients_dict)
                print(len(gradients_list))
            elif defense_method == "pop":
                print("pop defense (testing)")
                gradients_list = aggregation_rules.pop_worker_updates(gradients_dict, 0)
                # gradients_list = aggregation_rules.pop_one(gradients_list, 0)
                print(len(gradients_list))
            else:
                pass
                
            # gradients_list = list(gradients_dict.values())
            # aggregated_gradients = aggregation_rules.average_grads(gradients_list)
            aggregated_gradients, k = aggregator.aggregate(gradients_dict, worker_batch_map, curr_iter=iteration)
            corrected = aggregator.corrected

            # adjust learning rate: lr * sqrt(k)
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * np.sqrt(k)

            # Update global model
            for p, agg_grad in zip(global_model.parameters(), aggregated_gradients):
                p.grad = agg_grad
            optimizer.step()

            avg_iter_loss = sum(loss_list) / len(loss_list)
            epoch_loss_list.append(avg_iter_loss)
            # print(f'Epoch: {epoch+1}, sub-batch: {iteration}, avg sub-batch loss: {round(avg_iter_loss, 3)}')
            # print curr iter / total iter

            # infor
            print(f'EP: {epoch+1}/{n_epochs}, sub-batch: {iteration+1}/{len(data_loader)}, avg sub-batch loss: {round(avg_iter_loss, 3)}')
            # exit()
            # validation per 50% of the dataset, and at the last iteration
            if iteration % (len(data_loader) // 2) == 0 and iteration != 0 or iteration == len(data_loader) - 1:
            # if iteration % 30 == 0 and iteration != 0:
                acc = inference(global_model, test_loader, device)
                print("validation accuracy:", acc)
                # Write to tensorboard tag with worker number
                if tb_log_dir is not None:
                    writer.add_scalar('Accuracy/val', acc, training_iter)

            # Write to tensorboard tag with worker number
            if tb_log_dir is not None:
                writer.add_scalar('Loss/train', avg_iter_loss, training_iter)

        # Compute and print the average epoch loss
        avg_epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        # writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        print(f'Epoch: {epoch+1} done, avg loss: {round(avg_epoch_loss, 3)}')

    if tb_log_dir is not None: writer.close()
    