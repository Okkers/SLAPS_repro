from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic 
from models import SLAPS

import numpy as np 
import torch 
import torch.nn as nn
from torch import optim 

import os 
import time 

import warnings 

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "SLAPS": SLAPS
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
            # model = nn.DataParallel(model, device_ids = self.args.device_ids)
        return model 
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optimizer = 0 # SOME OPTIMIZER
        return model_optimizer
    
    def _select_metric(self):
        loss_classifier = nn.CrossEntropyLoss()

        if self.args.is_discrete:
            loss_gnn_dae = nn.BCELoss()
        else:
            loss_gnn_dae = nn.MSELoss()
        return loss_classifier, loss_gnn_dae
    
    def _predict(self, batch_x, batch_y):
        # MAKE PREDICTION
        # 
        # SHAPING AND PREPROCESSING

        outputs = self.model(batch_x)

        ### POSTPROCESSING

        # OUT
        return outputs, batch_y
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        ### IMPLEMENT VALIDATION FUNCTION
        return total_loss 
    
    def train(self, setting):
        ### IMPLEMENT TRAINING 


        return 
    
    def test(self, setting):
        ### IMPLEMENT TESTING


        return 
    
    def predict(self, setting):
        ### IMPLEMENT BATCH PREDICTION

        return 
    
    def bugfix(self):
        ### BUGFIXING NOISE FUNCTION FOR NOW 
        test_x = torch.ones((5,5))

        print(test_x)
        tilde_x, mask = self.model.add_noise(test_x)

        print(tilde_x)
        print(mask)
        return
