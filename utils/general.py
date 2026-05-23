import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, mode="min"):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode

    def _is_improvement(self, metric):
        if self.best_metric is None:
            return True

        if self.mode == "min":
            return metric < (self.best_metric - self.delta)

        return metric > (self.best_metric + self.delta)

    def __call__(self, metric, model, path):
        if self._is_improvement(metric):
            self.best_metric = metric
            self.save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path + '/checkpoint.pth')
