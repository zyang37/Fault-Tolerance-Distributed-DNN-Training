import torch
import numpy as np

# helper functions 
def flatten_grads(model_grads):
    # grads: list of gradients
    # grads[i]: gradients at i-th layers
    flat_grads_per_model = [torch.cat([g.view(-1) for g in grads], dim=0) for grads in model_grads]
    # concat flat_grads_per_model from all layers into a single vector
    flat_grads = torch.cat(flat_grads_per_model, dim=0)
    return flat_grads


def average_grads(gradients_list):
    # gradients_list: list of list of gradients
    # gradients_list[i]: list of gradients from i-th worker
    # gradients_list[i][j]: j-th gradient from i-th worker
    aggregated_gradients = []
    for i, grad in enumerate(zip(*gradients_list)):
        aggregated_gradients.append(sum(grad) / len(grad))
    return aggregated_gradients

def pop_one(grads, idx):
    grads.pop(idx)
    return grads

def mean_filter(gradients_list):
    worker_means = []
    for i, worker_grads in enumerate(gradients_list):
        worker_flat_grads = flatten_grads([worker_grads])
        worker_means.append(worker_flat_grads.mean())
        print(worker_flat_grads.mean())
    
    # compute diff between each worker's mean, and sum them up
    # var worker_dis_sum = [1, 2, 3,..]
    worker_dis_sum = []
    for i, worker_mean in enumerate(worker_means):
        worker_dis_sum.append(sum([abs(worker_mean - mean) for mean in worker_means]))

    # select the worker with the smallest sum
    _, selected_worker = torch.min(torch.tensor(worker_dis_sum), 0)

    return [gradients_list[selected_worker]]

# in progress
def krum(gradients_list, num_workers, f=1):
    """
    Krum algorithm for Byzantine-resilient gradient aggregation.

    Args:
        grads (List[Tensor]): A list of gradients from each worker.
        num_workers (int): Total number of workers.
        f (int): The number of Byzantine workers to tolerate.

    Returns:
        Tensor: The selected gradient after Krum aggregation.
    """
    # # Calculate pairwise distances
    # distances = torch.zeros((num_workers, num_workers))
    # for i in range(num_workers):
    #     for j in range(i + 1, num_workers):
    #         # gradients_list[i] is a list of gradients from i-th worker
    #         # distances[i, j] = torch.norm(gradients_list[i] - gradients_list[j])
    #         distances[i, j] = torch.norm(torch.stack(gradients_list[i]) - torch.stack(gradients_list[j]))
    #         distances[j, i] = distances[i, j]
    
    # # Compute scores
    # scores = torch.zeros(num_workers)
    # for i in range(num_workers):
    #     distances[i] = torch.sort(distances[i])[0]
    #     scores[i] = torch.sum(distances[i][:num_workers-f-1])

    # # Select the gradient with the minimum score
    # _, selected_worker = torch.min(scores, 0)
    # return [gradients_list[selected_worker]]

    return gradients_list
