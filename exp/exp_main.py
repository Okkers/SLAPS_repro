from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic 
from models import SLAPS_FP, SLAPS_MLP, SLAPS_MLPD
from utils.general import EarlyStopping
from utils.self_training import select_top_zeta_labels
from utils.ada_edge import apply_adj_processor, adjust_graph, get_class_plus_confidence

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

        self.adaedge_model = None
        self.adaedge_A = None

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
                {"params": gcnc_params, "lr": self.args.lr_c, "weight_decay": self.args.weight_decay_c},
                {"params": rest, "lr": self.args.lr_DAE, "weight_decay": self.args.weight_decay_dae},
            ]
        )

        return model_optimizer
    
    def _select_metric(self):
        loss_classifier = nn.CrossEntropyLoss()

        if self.args.is_discrete:
            loss_gnn_dae = nn.BCEWithLogitsLoss(reduction='none')
        else:
            loss_gnn_dae = nn.MSELoss(reduction='none')
        return loss_classifier, loss_gnn_dae
    
    def _predict(self, batch_x, batch_y):
        # MAKE PREDICTION
        # 
        # SHAPING AND PREPROCESSING

        outputs = self.model(batch_x)

        ### POSTPROCESSING

        # OUT
        return outputs, batch_y
    
    def _one_training_loop(self, features, train_mask, val_mask, labels, setting):
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

            # had to compute mean this way
            noise_mask = noise_mask.detach()
            loss_elementwise = loss_gnn_dae(x_reconstructed, features)

            if noise_mask.any():
                dae_loss = (loss_elementwise * noise_mask.float()).sum() / noise_mask.float().sum()
            else:
                dae_loss = loss_elementwise.mean()

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
    
    def _train_gnn_ada_edge(self, A, features, labels, train_mask, val_mask, test_mask, setting):
        from layers.GNN_C import GNN_C
        gnn_model = GNN_C(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.output_dim,
            dropout_c=self.args.dropout_c
        ).float().to(self.device)

        features = features.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        path = os.path.join(self.args.checkpoints, setting + "_ada_edge")
        if not os.path.exists(path):
            os.makedirs(path)

        use_cross_entropy = self.args.dataset in ["wine", "cancer"]
        early_stopping = EarlyStopping(patience = self.args.patience, mode="min" if use_cross_entropy else "max")

        gnn_model_optimizer = optim.Adam(
                gnn_model.parameters(), lr=self.args.lr_c
        )
        gnn_loss = nn.CrossEntropyLoss()

        log_every = 10

        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            gnn_model.train()
            gnn_model_optimizer.zero_grad()

            logits = gnn_model(A, features)

            loss = gnn_loss(logits[train_mask], labels[train_mask])
            loss.backward()
            gnn_model_optimizer.step()

            if (epoch + 1) % 100 == 0:
                    speed = (time.time() - time_now) / epoch
                    time_left = speed * ((self.args.train_epochs - epoch))
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, time_left))
                    time_now = time.time()


            if epoch % log_every == 0 or epoch == self.args.train_epochs - 1:
                gnn_model.eval()
                with torch.no_grad():
                    val_logits = gnn_model(A, features)
                    train_pred = torch.argmax(val_logits[train_mask], dim=1)
                    val_pred = torch.argmax(val_logits[val_mask], dim=1)

                    train_acc = (train_pred == labels[train_mask]).float().mean().item()

                    if use_cross_entropy:
                        val_metric = gnn_loss(val_logits[val_mask], labels[val_mask]).item()
                        monitor_name = "val_ce"
                    else:
                        val_metric = (val_pred == labels[val_mask]).float().mean().item()
                        monitor_name = "val_acc"

                    val_result = val_metric

                    early_stopping(val_metric, gnn_model, path)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break

                print(
                    f"Epoch [{epoch + 1}/{self.args.train_epochs}] "
                    f"loss={loss.item():.6f} "
                    f"train_acc={train_acc:.4f} "
                    f"{monitor_name}={val_metric:.4f}"
                )

        best_model_path = os.path.join(path, "checkpoint.pth")
        gnn_model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            
        gnn_model.eval()
        with torch.no_grad():
            val_logits = gnn_model(A, features)
            train_pred = torch.argmax(val_logits[train_mask], dim=1)
            val_pred = torch.argmax(val_logits[val_mask], dim=1)

            train_acc = (train_pred == labels[train_mask]).float().mean().item()

            if use_cross_entropy:
                val_metric = gnn_loss(val_logits[val_mask], labels[val_mask]).item()
                monitor_name = "val_ce"
            else:
                val_metric = (val_pred == labels[val_mask]).float().mean().item()
                monitor_name = "val_acc"

        return val_metric, gnn_model 

    def _ada_edge(self, features, labels, train_mask, val_mask, test_mask, setting):
        self.model.eval()
        with torch.no_grad():
            logits, _, _, adj = self.model(features, get_adj=True)
        
        adj = (adj > 0).float() # binarize adjacency matrix for ADA Edge
        
        val_pred = logits[val_mask].argmax(dim=1)
        initial_val_acc = (val_pred == labels[val_mask]).float().mean().item()
        best_val_acc = (val_pred == labels[val_mask]).float().mean().item()
        best_model = None

        print(f"Initial validation accuracy before ADA EDGE: {initial_val_acc:.4f}")

        for i in range(self.args.max_iter):
            print(f"ADA EDGE iteration {i+1}/{self.args.max_iter}")
            A = apply_adj_processor(adj)

            class_predictions, confidence, probabilities = get_class_plus_confidence(logits)

            A_new, added, removed = adjust_graph(
                A=adj,
                class_predictions=class_predictions,
                confidence=confidence,
                conf_threshold=self.args.conf_threshold,
                max_add=self.args.max_add,
                max_remove=self.args.max_remove,
                order=self.args.order
            )
            print(f"Added {added} edges, removed {removed} edges")

            A_norm = apply_adj_processor(A_new)
            val_acc, model = self._train_gnn_ada_edge(
                A=A_norm,
                features=features,
                labels=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                setting=setting
            )

            print(f"Validation accuracy after ADA EDGE iteration {i+1}: {val_acc:.4f}")

            if val_acc <= best_val_acc:
                print("Validation accuracy did not improve, stop ADA EDGE.")
                break

            best_val_acc = val_acc
            best_model = model
            adj = A_new.clone()

            best_model.eval()
            with torch.no_grad():
                logits = best_model(apply_adj_processor(adj), features)

        self.adaedge_model = best_model
        self.adaedge_A = apply_adj_processor(adj)

        print(f"Best validation accuracy after ADA EDGE: {best_val_acc:.4f} compared to initial {initial_val_acc:.4f}")


    def train(self, setting):
        features, labels, train_mask, val_mask, test_mask, f, c = self._get_data("train")
        features = features.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        print("Start initial SLAPS training loop")
        self._one_training_loop(features, train_mask, val_mask, labels, setting)

        if self.args.self_training:
            print("Start self-training")
            self.model.eval()
            with torch.no_grad():
                logits, _, _ = self.model(features)
                
            new_labels, new_train_mask, selected_nodes, selected_labels = select_top_zeta_labels(
                model=self.model,
                features=features,
                labels=labels,
                logits=logits,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                zeta=self.args.self_training_zeta
            )

            print(f"Original amount of training nodes: {train_mask.sum().item()}")
            print(f"New amount of training nodes: {new_train_mask.sum().item()}")

            self.model = self._build_model().float().to(self.device)

            print("Start training using the expanded set of labels")
            self._one_training_loop(
                features=features,
                labels=new_labels,
                train_mask=new_train_mask,
                val_mask=val_mask,
                setting=setting
            )

        if self.args.ada_edge:
            print("Start ADA EDGE:")
            self._ada_edge(
                features=features,
                labels=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                setting=setting
            )

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

        return test_acc
    
    def test_ada_edge(self, setting):
        features, labels, train_mask, val_mask, test_mask, f, c = self._get_data("test")
        features = features.to(self.device)
        labels = labels.to(self.device)
        test_mask = test_mask.to(self.device)

        if self.adaedge_model is None:
            return self.test(setting)
        
        self.adaedge_model.eval()
        with torch.no_grad():
            logits = self.adaedge_model(self.adaedge_A.to(self.device), features)
            test_pred = torch.argmax(logits[test_mask], dim=1)
            test_acc = (test_pred == labels[test_mask]).float().mean().item()

        print(f"ADA EDGE Test Accuracy: {test_acc:.4f}")

        f = open("result.txt", "a")
        f.write(setting + "  \n")
        f.write(f"Accuracy: {test_acc}")
        f.write("\n")
        f.write("\n")
        f.close()

        return test_acc
    
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
