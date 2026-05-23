import torch.nn as nn
import torch.nn.functional as F


class GNN_DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=None, dropout_adj=0.25):
        """
        This will be called with A (output from adj_prep) and X_tilde (noisy version of node features).

        :param input_dim: num of features, taken from X.shape[1]
        :param hidden_dim: usually 32, but one dataset (ogbn-arxiv) 256, part B in Supplement
        :param output_dim: feature reconstruction dimension
        :param dropout_adj: was tuned {0.25, 0.5}, to reproduce use params from Table 1 in Supplementary
        """

        super(GNN_DAE, self).__init__()

        self.W_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.dropout_feat = nn.Dropout(p=0.5)  # fixed at 0.5, applied after layer 1, so not train
        self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, A, X):
        # Dropout to adj matrix
        A = self.dropout_adj(A)

        # Layer 1: ReLU(A * W_1)
        L_1 = F.relu(self.W_1(A @ X))
        # Dropout after first layer
        L_1 = self.dropout_feat(L_1)

        # Layer 2: logits = A * L_1 * W2
        X_denoised = self.W_2(A @ L_1)

        return X_denoised

    def loss_c(self, A, X, labels):
        logits = self.forward(A, X)
        return F.cross_entropy(logits, labels)