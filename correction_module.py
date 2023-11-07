'''
This file contains correction module for DDP.
- data collection
- Training correction models (from torch and sklearn)
- inference using correction models

Notes: idealy, this should run on a separate process, to avoid blocking the main process
'''

import torch
from torch.utils.tensorboard import SummaryWriter

# a class that includes all mechanisms for correction
class CorrectionModule:
    def __init__(self, tb_log_dir=None, device="cpu"):
        self.tb_log_dir = tb_log_dir
        self.device = torch.device("cpu")
        self.writer = None
        if tb_log_dir is not None:
            self.writer = SummaryWriter(log_dir=tb_log_dir)
        self.gradient_dataset =[]

    def collect_data(self):
        pass

    def train_correction_model(self):
        pass

    def correct_gradients(self):
        pass
