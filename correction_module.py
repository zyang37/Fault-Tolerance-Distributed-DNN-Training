'''
This file contains correction module for DDP.
- data collection
- Training correction models (from torch and sklearn)
- inference using correction models

Idealy, this should run on a separate process, to avoid blocking the main process
'''

import os
import torch
import pickle
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from torch.utils.tensorboard import SummaryWriter

seed_value = 1

class CorrectionModule:
    '''
    This class that includes all mechanisms for correction
    It is initialized in aggregation_rules.py
    '''
    def __init__(self, worker_id, train_at_iter=1, model=None, tb_log_dir=None, device="cpu", data_coll_path=False):
        '''
        Note: 
            - data_coll_path: if is a [path], the correction module will only collect data, and not train the correction model
        '''
        self.worker_id = worker_id
        self.model = model
        self.train_at_iter = train_at_iter
        self.model_status = "not initailized"
        self.gradient_dataset ={'input':[], 'target':[]}
        self.device = torch.device(device)
        self.tb_log_dir = tb_log_dir
        self.writer = None
        self.data_coll_path = data_coll_path

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
        if self.dataset_size() >= self.train_at_iter:
            if self.data_coll_path:
                print("[CORRECTION] Data collected at {} iterations, EXIT!".format(self.dataset_size()))
                self.write_dataset_to_disk(self.data_coll_path)
                exit()
            else:
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
        # if path (only the dir part) not exist, create it
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

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

        # TODO: need update here!!! + std step
        input_grads = self.gradient_dataset['input'][0].reshape(-1, 1)
        target_grads = self.gradient_dataset['target'][0].reshape(-1, 1)
        # print(input_grads[0].shape, target_grads[0].shape)
        
        # std the data
        data_coverage = 1
        inputs_std, targets_std, input_scaler, target_scaler = data_standardization(input_grads, target_grads)
        # input_grads, target_grads = sample_data([inputs_std, targets_std], percent=data_coverage)

        self.model.fit(inputs_std, targets_std)
        self.model_status = "trained"
        
        # save training loss (MSE)
        preds = inference(input_grads, self.model, input_scaler, target_scaler)
        self.train_loss_mse = mean_squared_error(target_grads, preds)
        self.train_loss_mape = mean_absolute_percentage_error(target_grads+1, preds+1)
        # self.train_loss = ((self.model.predict(input_grads) - target_grads) ** 2).mean()
        # print("[CORRECTION] Training loss (MSE):", self.train_loss)
        print("[CORRECTION] Training loss (MSE, MAPE):", self.train_loss_mse, self.train_loss_mape)

        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        # empty the dataset
        self.gradient_dataset ={'input':[], 'target':[]}

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
        # corrected_flat_grads = self.model.predict(input_flat_grads.reshape(-1, 1))
        corrected_flat_grads = inference(input_flat_grads, self.model, self.input_scaler, self.target_scaler)

        # post processing
        # weighted
        corrected_flat_grads = corrected_flat_grads * 0.95
        corrected_grads_list = reconstruct_grads(torch.tensor(corrected_flat_grads), model)
        return corrected_grads_list



# Helper functions
def sample_data(datalist, sample_size=None, percent=None):
    '''
    Sample the data from the datalist
    args:
        - datalist: [data1, data2, ...]
        - data1: numpy array of shape (num_samples, num_features)
    '''
    if sample_size is None and percent is None:
        raise ValueError("Either sample_size or percent must be specified.")
    elif sample_size is None and percent is not None:
        # calculate the sample size
        sample_size = int(datalist[0].shape[0] * percent)
        # print(sample_size)

    indices = np.random.choice(datalist[0].shape[0], sample_size, replace=False)
    return [data[indices] for data in datalist]

def data_standardization(inputs, targets):
    '''
    Standardize the data
    args:
        - datalist: [data1, data2, ...]
        - data1: numpy array of shape (num_samples, num_features)
    '''
    # standardize the dataset
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    input_scaler.fit(inputs)
    inputs_std = input_scaler.transform(inputs)

    target_scaler.fit(targets)
    targets_std = target_scaler.transform(targets)

    # return inputs_std, targets, input_scaler#, target_scaler
    return inputs_std, targets_std, input_scaler, target_scaler

def data_destandardization(data, scaler):
    '''
    De-standardize the data
    args:
        - data: numpy array of shape (num_samples, num_features)
        - scaler: sklearn.preprocessing.StandardScaler
    '''
    return scaler.inverse_transform(data.reshape(-1, 1))

def inference(input_grads, model, input_scaler, target_scaler):
    '''
    Inference the model
    args:
        - grad_list: list of gradients
        - model: model
        - input_scaler: sklearn.preprocessing.StandardScaler
        - target_scaler: sklearn.preprocessing.StandardScaler
    '''
    if len(input_grads.shape) == 1:
        input_grads = input_grads.reshape(-1, 1)

    # standardize the data
    input_grads_std = input_scaler.transform(input_grads)
    # inference
    pred_std = model.predict(input_grads_std)
    # de-standardize the data
    # pred_grads = pred_std.reshape(-1, 1)
    pred_grads = target_scaler.inverse_transform(pred_std.reshape(-1, 1))
    return pred_grads

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
