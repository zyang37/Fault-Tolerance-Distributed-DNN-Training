'''
This file contains aggregation rules (faulty gradient detection algorithms) for DDP.
'''

import torch
import numpy as np
from correction_module import CorrectionModule as CM

seed_value = 1

class Aggregator:
    '''
    This class that includes all mechanisms for aggregation, and handling communication with the correction module
    '''
    def __init__(self, tb_log_dir=None, device="cpu", correct_args={}):
        
        self.correct = correct_args['correct']
        self.cmodel = correct_args['cmodel']
        self.model = correct_args['model']
        self.data_coll_path = correct_args['data_coll_path']
        self.corrected = False
        
        self.device = torch.device(device)
        self.tb_log_dir = tb_log_dir
        self.faulty_worker_idxs = []
        self.correction_models = {}

    def aggregate(self, gradients_dict, worker_batch_map, verbose=False):
        '''
        Aggregate gradients from gradients_dict (avg)
        '''
        self.gradients_dict = gradients_dict
        self.worker_batch_map = worker_batch_map

        self.detect_faulty_gradients()

        if len(self.faulty_worker_idxs) > 0:
            self.handle_faulty_gradients()

        if verbose:
            print("[AGG] used gradients from workers: {}".format(sorted(list(self.gradients_dict.keys()))))
        gradients_list = list(self.gradients_dict.values())
        aggregated_gradients = average_grads(gradients_list)
        return aggregated_gradients, len(gradients_list)

    def update_faulty_worker_idxs(self, faulty_worker_idxs):
        '''
        Update faulty_worker_idxs
        '''
        self.faulty_worker_idxs = faulty_worker_idxs

    def detect_faulty_gradients(self):
        '''
        Detect faulty gradients from gradients_dict and update faulty_worker_idxs
        '''
        # TODO: implement faulty gradient detection
        if len(self.faulty_worker_idxs)==0:
            # detect
            # update self.faulty_worker_idxs
            pass
        else:
            pass
    
    def handle_faulty_gradients(self):
        '''
        Handle faulty gradients from gradients_list
        '''
        if self.correct:
            if len(self.correction_models) == 0:
                self.correction_model_init()
            self.correct_faulty_gradients()
        else:
            self.remove_faulty_gradients()

    def correct_faulty_gradients(self):
        '''
        Correct faulty gradients from gradients_dist. 
        loop through correction_models:
            if model not trained, add data, remove faulty gradients
            else, correct faulty gradients
        '''
        for faulty_id, correction_model in self.correction_models.items():
            if correction_model.model_status != "trained":
                # add data, if will train if the dataset is large enough
                correction_model.collect_data(self.gradients_dict[faulty_id], self.gradients_dict[self.worker_batch_map[faulty_id]])
                # remove faulty gradients
                del self.gradients_dict[faulty_id]
            else:
                self.corrected = True
                # correct faulty gradients
                self.gradients_dict[faulty_id] = correction_model.correct_gradients(self.gradients_dict[faulty_id], self.model)
    
    def correction_model_init(self):
        '''
        init correction models, based on faulty_worker_idxs and worker_batch_map
        '''
        for i in self.faulty_worker_idxs:
            print("[AGG] Init correction model for worker {}".format(i))
            # self.correction_models[i] = CM(i, self.cmodel, self.tb_log_dir, self.device)
            self.correction_models[i] = CM(worker_id=i, 
                                           train_at_iter=10, 
                                           model=self.cmodel, 
                                           tb_log_dir=self.tb_log_dir, 
                                           device="cpu", 
                                           data_coll_path=self.data_coll_path)

    def remove_faulty_gradients(self):
        '''
        Remove faulty gradients from gradients_dist
        '''
        for idx in self.faulty_worker_idxs:
            del self.gradients_dict[idx]
            


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

def pop_worker_updates(gradients_dict, idx):
    # testing, remove key at idx
    del gradients_dict[idx]
    return gradients_dict

# in progress
def mean_filter(gradients_dict, f=1):
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

    # find top k max
    _, de_selected_worker = torch.topk(torch.tensor(worker_dis_sum), k=f, largest=False)
    print(de_selected_worker)
    filtered_grads = []
    for i in range(len(gradients_list)):
        if i != de_selected_worker:
            filtered_grads.append(gradients_list[i])

    # return [gradients_list[selected_worker]]
    return filtered_grads

# in progress
def krum(gradients_dict, num_workers, f=1):
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
    return gradients_dict
