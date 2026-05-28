from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch.nn.functional as F
import torch

def initialize_kNN_graph(X, k):  
    X = F.normalize(X, p=2, dim=1) # Normalize the features to unit length
    S = torch.mm(X, X.t()) # Compute the cosine similarity matrix; since we normalized, it is just the dot product

    S_mask = S.clone() # exclude self loop such that we get self + k neighbours
    S_mask.fill_diagonal_(-float('inf'))

    topk = torch.topk(S_mask, k=k, dim=-1)
    A = torch.zeros_like(S)
    A.fill_diagonal_(1.0)
    A.scatter_(dim=1, index=topk.indices, value=1.0)

    A = A * 10.0 - 10.0 

    return A


def kNN_generator(X,k):
    # This is exactly M dot S described in the paper, but gradients will not flow:
    # M_S = kneighbors_graph(X=X, n_neighbors=k+1, mode='distance', metric='cosine', include_self=True) 

    X = F.normalize(X, p=2, dim=1) # Normalize the features to unit length
    S = torch.mm(X, X.t()) # Compute the cosine similarity matrix; since we normalized, it is just the dot product

    S_mask = S.clone() # exclude self loop such that we get self + k neighbours
    S_mask.fill_diagonal_(-float('inf'))

    topk = torch.topk(S, k=k, dim=-1)
    M = torch.zeros_like(S)
    M.fill_diagonal_(1.0)
    M.scatter_(dim=1, index=topk.indices, value=1.0)

    M_S = M * S
    return M_S