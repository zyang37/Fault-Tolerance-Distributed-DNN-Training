import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Manager
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod

def noisy_gradients_gen(torch_model, mode: int, noise_type: int, noise_scale: float):
    '''
    args:
        - torch_model: the model grad to be altered
        - mode: 0 deterministic, 1 add random, 2 random
        - noise_type: For different mode, noise_type can include different noise types
        - noise_scale: the scale of noise
    return: 
        - a list of altered gradients
    '''
    # gradients = [p.grad.clone() for p in self.model.parameters()]
    if mode==0:
        # deterministic
        pass
    elif mode==1:
        # add random
        pass
    elif mode==2:
        # generate random gradients that matches the shape of the model gradients
        altered_gradients = [torch.randn_like(p.grad.clone()) for p in torch_model.parameters()]
    else:
        raise NotImplementedError
    
    return altered_gradients


class Worker:
    def __init__(self, optimizer, model) -> None:
        self.model = model
        self.optimizer = optimizer.zero_grad

    
    def action(self, target, gradients_list, loss_list, data):
        output = self.model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        gradients = [p.grad.clone() for p in self.model.parameters()]
        gradients_list.append(gradients)
        loss_list.append(loss.item())

    @abstractmethod
    def get_gradients(self, model):
        pass


class HealthyWorker(Worker):
    def get_gradients(self):
        return [p.grad.clone() for p in self.model.parameters()]
    

class FaultyWorker(Worker):
    def get_gradients(self):
        return [torch.ones_like(p.grad.clone()) * 1e9 for p in self.model.parameters()]