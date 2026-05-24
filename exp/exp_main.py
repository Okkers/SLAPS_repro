from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic 
from models import SLAPS_FP, SLAPS_MLP, SLAPS_MLPD
from utils.general import EarlyStopping

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
            "SLAPS_FP": SLAPS_FP,
            "SLAPS_MLP": SLAPS_MLP,
            "SLAPS_MLPD": SLAPS_MLPD
        }
        features, labels, train_mask, val_mask, test_mask, f, c = self._get_data("train")

        self.args.features = features

        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
            # model = nn.DataParallel(model, device_ids = self.args.device_ids)
        return model 
    
    def _get_data(self, flag):
        return data_provider(self.args)
    
    def _select_optimizer(self):
        gcnc_params = list(self.model.gcn_c.parameters())
        gcnc_ids = {id(p) for p in gcnc_params}
        rest = [p for p in self.model.parameters() if id(p) not in gcnc_ids]

        # Keep learning rates disparate between GNN_C params and the rest  
        model_optimizer = optim.Adam(
            [
                {"params": gcnc_params, "lr": self.args.lr_c},
                {"params": rest, "lr": self.args.lr_DAE}
            ]
        )

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
    

    def train(self, setting):
        features, labels, train_mask, val_mask, test_mask, f, c = self._get_data("train")
        features = features.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        use_cross_entropy = self.args.dataset in ["wine", "cancer"]
        early_stopping = EarlyStopping(patience = self.args.patience, mode="min" if use_cross_entropy else "max")


        model_optim = self._select_optimizer()
        loss_clf, loss_gnn_dae = self._select_metric()

        log_every = 10

        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            model_optim.zero_grad()

            logits_c, x_reconstructed, noise_mask = self.model(features)

            class_loss = loss_clf(logits_c[train_mask], labels[train_mask])

            if noise_mask.dtype != torch.bool:
                noise_mask = noise_mask.bool()
            if noise_mask.any():
                dae_loss = loss_gnn_dae(x_reconstructed[noise_mask], features[noise_mask])
            else:
                dae_loss = loss_gnn_dae(x_reconstructed, features)

            total_loss = class_loss + self.args.lambda_val * dae_loss
            total_loss.backward()
            model_optim.step()

            if (epoch + 1) % 100 == 0:
                    speed = (time.time() - time_now) / epoch
                    time_left = speed * ((self.args.train_epochs - epoch))
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, time_left))
                    time_now = time.time()


            if epoch % log_every == 0 or epoch == self.args.train_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _, _ = self.model(features)
                    train_pred = torch.argmax(logits_c[train_mask], dim=1)
                    val_pred = torch.argmax(val_logits[val_mask], dim=1)

                    train_acc = (train_pred == labels[train_mask]).float().mean().item()

                    if use_cross_entropy:
                        val_metric = loss_clf(val_logits[val_mask], labels[val_mask]).item()
                        monitor_name = "val_ce"
                    else:
                        val_metric = (val_pred == labels[val_mask]).float().mean().item()
                        monitor_name = "val_acc"

                    early_stopping(val_metric, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break

                print(
                    f"Epoch [{epoch + 1}/{self.args.train_epochs}] "
                    f"loss={total_loss.item():.6f} "
                    f"class={class_loss.item():.6f} "
                    f"dae={dae_loss.item():.6f} "
                    f"train_acc={train_acc:.4f} "
                    f"{monitor_name}={val_metric:.4f}"
                )
            


        best_model_path = path + "/" + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return
    
    def test(self, setting):
        features, labels, train_mask, val_mask, test_mask, f, c = self._get_data("test")
        features = features.to(self.device)
        labels = labels.to(self.device)
        test_mask = test_mask.to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, "checkpoint.pth")


        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        with torch.no_grad():
            logits_c, _, _ = self.model(features)
            test_pred = torch.argmax(logits_c[test_mask], dim=1)
            test_acc = (test_pred == labels[test_mask]).float().mean().item()

        print(f"Test Accuracy: {test_acc:.4f}")

        f = open("result.txt", "a")
        f.write(setting + "  \n")
        f.write(f"Accuracy: {test_acc}")
        f.write("\n")
        f.write("\n")
        f.close()

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
