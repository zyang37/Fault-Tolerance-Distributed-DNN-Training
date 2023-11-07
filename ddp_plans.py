'''
This file contains DDP strategies for distributing data and target batches to workers.
Mainly when there are faulty workers, we distribute the data and target batches replicas. 
'''

import random

# Divide a tensor into a specific number of sub-batches
def divide_into_sub_batches(tensor, num_sub_batches):
    sub_batch_size = tensor.size(0) // num_sub_batches
    return [tensor[i:i + sub_batch_size] for i in range(0, tensor.size(0), sub_batch_size)]

# distribute a batch of data and target into sub-batches (no faulty workers)
def distribute_subbatch(data_batch, target_batch, num_workers):
    data_sub_batches = divide_into_sub_batches(data_batch, num_workers)
    target_sub_batches = divide_into_sub_batches(target_batch, num_workers)
    return data_sub_batches, target_sub_batches

# A gereralized version of distribute_subbatch
# distribute a batch of data and target into sub-batches (with faulty workers)
def distribute_subbatch_general(data_batch, target_batch, num_workers: int, faulty_worker_ids: list):
    # in this function, we will randomly select a healthy worker to replace the faulty worker
    # return the idx of the healthy worker id, and the modified data and target sub-batches

    # raise warning if the number of faulty workers is greater than the number of healthy workers
    if len(faulty_worker_ids) > num_workers - len(faulty_worker_ids):
        print("Warning: the number of faulty workers is greater than the number of healthy workers")

    data_sub_batches = divide_into_sub_batches(data_batch, num_workers)
    target_sub_batches = divide_into_sub_batches(target_batch, num_workers)
    healthy_worker_idx = [i for i in range(0, num_workers) if i not in faulty_worker_ids]

    modified_data_sub_batch = []
    modified_target_sub_batch = []
    random_healthy_idx = random.choice(healthy_worker_idx)
    for faulty_idx in faulty_worker_ids:
        modified_data_sub_batch[faulty_idx] = data_sub_batches[random_healthy_idx]
        modified_target_sub_batch[faulty_idx] = target_sub_batches[random_healthy_idx]
    
    return modified_data_sub_batch, modified_target_sub_batch, random_healthy_idx


class DDP:
    def __init__(self):
        # TODO, add some stuff later
        pass

    def divide_into_sub_batches(self, tensor, num_sub_batches):
        sub_batch_size = tensor.size(0) // num_sub_batches
        return [tensor[i:i + sub_batch_size] for i in range(0, tensor.size(0), sub_batch_size)]

    def distribute_batch(self, data_batch, target_batch, num_workers, faulty_worker_idx):
        # TODO: Check with worker class to see how each worker's data-batches are stored
        # faulty_worker.set_batches(healthy_worker.get_batches())

        data_sub_batches = self.divide_into_sub_batches(data_batch, num_workers)
        target_sub_batches = self.divide_into_sub_batches(target_batch, num_workers)

        modified_data_sub_batch = []
        modified_target_sub_batch = []

        healthy_worker_idx = [i for i in range(0, num_workers) if i not in faulty_worker_idx]
        for faulty_idx in faulty_worker_idx:
            random_healthy_idx = random.choice(healthy_worker_idx)
            modified_data_sub_batch[faulty_idx] = data_sub_batches[random_healthy_idx]
            modified_target_sub_batch[faulty_idx] = target_sub_batches[random_healthy_idx]

        return modified_data_sub_batch, modified_target_sub_batch


