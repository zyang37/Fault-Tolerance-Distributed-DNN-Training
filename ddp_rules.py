import random


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


