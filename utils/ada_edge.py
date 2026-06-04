import torch
import torch.nn.functional as F
from layers.GNN_C import GNN_C

def get_class_plus_confidence(logits):
    """Returns the class predictions with confidence and probabilities."""
    probabilities = F.softmax(logits, dim=1)
    confidence, class_predictions = probabilities.max(dim=1)
    return class_predictions, confidence, probabilities

def apply_adj_processor(A):
    A = A.clone()
    degrees = A.sum(dim=1)
    D_inv_sqrt = torch.where(degrees > 0, degrees.pow(-0.5), 0.0) # to avoid division by 0

    A = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)

    return A

def add_edges(A, class_predictions, confidence, conf_threshold, max_add):
    A_add = A.clone()

    total_added = 0

    # We only consider nodes where confidence of being in class C is >= conf_threshold
    confident_nodes = torch.where(confidence >= conf_threshold)[0].tolist()

    for i in range(len(confident_nodes)):
        node1 = confident_nodes[i]

        for j in range(i+1, len(confident_nodes)):
            node2 = confident_nodes[j]

            # Check if there is already an edge between node1 and node2
            if A_add[node1, node2] != 0 or A_add[node2, node1] != 0:
                continue

            # Check if the classes are corresponding:
            if class_predictions[node1] != class_predictions[node2]:
                continue

            # Add undirected edge:
            A_add[node1, node2] = 1.0
            A_add[node2, node1] = 1.0

            total_added+=1

            if total_added >= max_add:
                return A_add, total_added
    
    return A_add, total_added

def remove_edges(A, class_predictions, confidence, conf_threshold, max_remove):
    A_remove = A.clone()

    total_removed = 0

    # We only consider nodes where confidence of being in class C is >= conf_threshold
    confident_nodes = torch.where(confidence >= conf_threshold)[0].tolist()

    for i in range(len(confident_nodes)):
        node1 = confident_nodes[i]

        for j in range(i+1, len(confident_nodes)):
            node2 = confident_nodes[j]

            # Check if there is an edge between node1 and node2
            if A_remove[node1, node2] == 0 and A_remove[node2, node1] == 0:
                continue

            # Check if the classes are corresponding:
            if class_predictions[node1] == class_predictions[node2]:
                continue

            # Add undirected edge:
            A_remove[node1, node2] = 0.0
            A_remove[node2, node1] = 0.0

            total_removed+=1

            if total_removed >= max_remove:
                return A_remove, total_removed
    
    return A_remove, total_removed

def adjust_graph(A, class_predictions, confidence, conf_threshold, max_add, max_remove, order):
    if order == "add":
        A_new, added = add_edges(A, class_predictions, confidence, conf_threshold, max_add)
        A_new, removed = remove_edges(A_new, class_predictions, confidence, conf_threshold, max_remove)
    else:
        A_new, removed = remove_edges(A, class_predictions, confidence, conf_threshold, max_remove)
        A_new, added = add_edges(A_new, class_predictions, confidence, conf_threshold, max_add)
    return A_new, added, removed