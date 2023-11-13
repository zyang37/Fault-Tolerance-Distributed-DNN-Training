'''
This file contains DDP strategies for distributing data and target batches to workers.
Mainly when there are faulty workers, we distribute the data and target batches replicas. 
'''

import random

class DDPDataDistributor:
    def __init__(self, num_workers: int, faulty_worker_ids: list = [], correction: bool = False):
        self.num_workers = num_workers
        self.faulty_worker_ids = faulty_worker_ids
        self.correct = correction
        self.worker_batch_map = {} # map worker id to batch id. Ex: {worker_0: batch_0, worker_1: batch_1, faulty_worker_2: batch_0}
        # init mapping
        for i in range(self.num_workers):
            # when there is no faulty workers, the mapping is n:n
            self.worker_batch_map[i] = i

    def distribute(self, data_batch, target_batch, corrected=False):
        '''
        Distribute data and target batches to workers
        '''
        if self.correct and len(self.faulty_worker_ids)>0 and corrected==False:
            # if there are faulty workers, we need to distribute the data and target batches replicas
            data_sub_batches, target_sub_batches, self.worker_batch_map = self.alter_distribute_subbatch(data_batch, target_batch)
        else:
            data_sub_batches, target_sub_batches = self.distribute_subbatch(data_batch, target_batch)
        return data_sub_batches, target_sub_batches, self.worker_batch_map

    def add_faulty_worker(self, worker_id):
        self.faulty_worker_ids.append(worker_id)

    def alter_distribute_subbatch(self, data_batch, target_batch):
        '''
        A gereralized version of distribute_subbatch
        distribute a batch of data and target into sub-batches (with faulty workers)

        in this function, we will randomly select a healthy worker to replace the faulty worker
        return the idx of the healthy worker id, and the modified data and target sub-batches
        '''
        if len(self.faulty_worker_ids) > self.num_workers - len(self.faulty_worker_ids):
            # raise warning if the number of faulty workers is greater than the number of healthy workers
            print("Warning: the number of faulty workers is greater than the number of healthy workers")

        data_sub_batches = self.divide_into_sub_batches(data_batch, self.num_workers)
        target_sub_batches = self.divide_into_sub_batches(target_batch, self.num_workers)
        healthy_worker_idx = [i for i in range(0, self.num_workers) if i not in self.faulty_worker_ids]

        modified_data_sub_batch = []
        modified_target_sub_batch = []
        
        # no matter how many faulty workers, we only need to randomly select one healthy worker
        random_healthy_idx = random.choice(healthy_worker_idx)
        for faulty_idx in self.faulty_worker_ids:
            # print(faulty_idx, random_healthy_idx)
            data_sub_batches[faulty_idx] = data_sub_batches[random_healthy_idx]
            target_sub_batches[faulty_idx] = target_sub_batches[random_healthy_idx]
            # update mapping
            self.worker_batch_map[faulty_idx] = random_healthy_idx
        
        return data_sub_batches, target_sub_batches, self.worker_batch_map
    
    def distribute_subbatch(self, data_batch, target_batch):
        # distribute a batch of data and target into sub-batches (no faulty workers)
        data_sub_batches = self.divide_into_sub_batches(data_batch, self.num_workers)
        target_sub_batches = self.divide_into_sub_batches(target_batch, self.num_workers)

        # update mapping
        for i in range(self.num_workers):
            self.worker_batch_map[i] = i

        return data_sub_batches, target_sub_batches

    @staticmethod
    def divide_into_sub_batches(tensor, num_sub_batches):
        # Divide a tensor into a specific number of sub-batches
        sub_batch_size = tensor.size(0) // num_sub_batches
        return [tensor[i:i + sub_batch_size] for i in range(0, tensor.size(0), sub_batch_size)]
    