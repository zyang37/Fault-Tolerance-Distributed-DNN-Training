'''
This file contains correction module for DDP.
- data collection
- Training correction models (from torch and sklearn)
- inference using correction models

Idealy, this should run on a separate process, to avoid blocking the main process
'''

import torch
import pickle
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from torch.utils.tensorboard import SummaryWriter

seed_value = 1


class CorrectionModule:
    '''
    This class that includes all mechanisms for correction
    '''
    def __init__(self, worker_id, model=None, tb_log_dir=None, device="cpu"):
        self.worker_id = worker_id
        self.model = model
        self.model_status = "not initailized"
        self.gradient_dataset ={'input':[], 'target':[]}
        self.device = torch.device(device)
        self.tb_log_dir = tb_log_dir
        self.writer = None

        if model is not None:
            self.regarsion_model_init(model)

        if tb_log_dir is not None:
            self.writer = SummaryWriter(log_dir=tb_log_dir)

    def collect_data(self, input_grads, target_grads):
        '''
        Add input_grads, and target_grads to the dataset. 
        AND check if the dataset is large enough to train the correction model
        args:
            - input_grads: flattened input gradients
            - target_grads: flattened target gradients
        '''
        # flatten the gradients
        flat_input_grads = self.data_preprocessing(input_grads)
        flat_target_grads = self.data_preprocessing(target_grads)
        # print(flat_input_grads.shape, flat_target_grads.shape)

        self.gradient_dataset['input'].append(flat_input_grads)
        self.gradient_dataset['target'].append(flat_target_grads)

        # check if the dataset is large enough to train the correction model
        # TODO: add a threshold here!!!
        if self.dataset_size() >= 1:
            print("[CORRECTION] Training ...")
            self.train_model()

    def dataset_size(self):
        '''
        Return the size of the gradient_dataset
        '''
        return len(self.gradient_dataset['input'])

    def write_dataset_to_disk(self, path):
        '''
        Save the gradient_dataset to disk as a pickle file (name.pkl)
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.gradient_dataset, f)

    def load_dataset_from_disk(self, path):
        '''
        Load the gradient_dataset from disk as a pickle file (name.pkl)
        '''
        with open(path, 'rb') as f:
            self.gradient_dataset = pickle.load(f)

    def regarsion_model_init(self, model:str):
        '''
        Init a regression model, selected from a list of sklearn models
        '''
        self.model_status = "initailized"
        if model == "svm":
            self.model = svm.SVR()
        elif model == "linear":
            self.model =  linear_model.LinearRegression()
        elif model == "mlp":
            self.model =  MLPRegressor(random_state=seed_value, max_iter=500)
        else:
            self.model_status = "not initailized"
            raise ValueError("Model not supported")

    def train_model(self):
        '''
        Train the correction model using the gradient_dataset
        '''
        if self.model is None:
            raise ValueError("Model not initialized")
        input_grads = self.gradient_dataset['input'][0].reshape(-1, 1)
        target_grads = self.gradient_dataset['target'][0].reshape(-1, 1)
        # print(input_grads[0].shape, target_grads[0].shape)
        # exit()

        self.model.fit(input_grads, target_grads)
        self.model_status = "trained"
        
        # save training loss (MSE)
        self.train_loss = ((self.model.predict(input_grads) - target_grads) ** 2).mean()
        print("[CORRECTION] Training loss (MSE):", self.train_loss)

        # TODO: add MSE threshold here!!! (if MSE is too high, we should not use this model)

        # if self.writer is not None:
        #     self.writer.add_scalar('Loss/train', self.model.loss_, self.dataset_size())

    def data_preprocessing(self, gradients_list:list):
        '''
        Pre-processing the gradients before adding to the dataset
        '''
        flat_grads = flatten_grads(gradients_list)
        # tensors should be on cpu, but just in case
        return flat_grads.detach().cpu().numpy()

    def correct_gradients(self, gradients_list:list, model):
        # check model status
        if self.model_status != "trained":
            raise ValueError("Model not trained")
        
        # flatten gradients, and pre-processing
        input_flat_grads = self.data_preprocessing(gradients_list)
        corrected_flat_grads = self.model.predict(input_flat_grads.reshape(-1, 1))

        # post processing
        # weighted
        corrected_flat_grads = corrected_flat_grads * 0.95
        corrected_grads_list = reconstruct_grads(torch.tensor(corrected_flat_grads), model)
        return corrected_grads_list



# Helper functions
def inspect_model_grads(gradients_list:list):
    '''
    This function is used for debugging purpose, print out the shape of each gradient

    args:
        - gradients_list: list of gradients (grads for diffent layers)
    '''
    for layer_grads in gradients_list:
        print(layer_grads.shape)

def flatten_grads(gradients_list:list):
    '''
    Example: 
        A list of gradients looks like this:
            torch.Size([512, 784])
            torch.Size([512])
            torch.Size([128, 512])
            torch.Size([128])
            torch.Size([10, 128])
            torch.Size([10])
        return
            torch.Size([468874])

    args:
        - gradients_list: list of gradients (grads for diffent layers)
    return:
        - flat_grads: flattened gradients
    '''
    # grads[i]: gradients at i-th layers
    flat_grads_per_model = [torch.cat([g.view(-1) for g in grads], dim=0) for grads in gradients_list]
    # concat flat_grads_per_model from all layers into a single vector
    flat_grads = torch.cat(flat_grads_per_model, dim=0)
    return flat_grads

def reconstruct_grads(flat_grads, model):
    '''
    Take in a flattened vector, and reconstruct the original gradients

    Example: 
        torch.Size([468874])
        
        return a list of gradients matching the model structure:
            [torch.Size([512, 784])
            torch.Size([512])
            torch.Size([128, 512])
            torch.Size([128])
            torch.Size([10, 128])
            torch.Size([10])]

    args:
        - flat_grads: flattened gradients
        - model: the model that we want to reconstruct the gradients
    return:
        - gradients_list: list of gradients (grads for diffent layers)
    '''
    # reconstruct the gradients
    gradients_list = []
    start_idx = 0
    for param in model.parameters():
        end_idx = start_idx + param.numel()
        gradients_list.append(flat_grads[start_idx:end_idx].view(param.shape))
        start_idx = end_idx
    return gradients_list
