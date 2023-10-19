import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Manager
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod


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