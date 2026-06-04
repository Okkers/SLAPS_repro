import torch
import torch.nn.functional as F

def get_available_masks(train_mask, val_mask, test_mask):
    """Returns the nodes that are still available, i.e., that are not already train, validation or test."""
    return (~train_mask) & (~val_mask) & (~test_mask)

def select_top_zeta_labels(model, features, labels, logits, train_mask, val_mask, test_mask, zeta):
    """Selects top zeta most confident predictions."""
    probabilities = F.softmax(logits, dim=1)

    available_masks = get_available_masks(train_mask, val_mask, test_mask)
    available_idx = torch.where(available_masks)[0]

    if available_idx.shape[0] == 0: # then no available nodes left 
        empty = torch.empty(0)
        return labels.clone(), train_mask.clone(), empty, empty
    
    candidate_probabilities = probabilities[available_idx]
    confidence, class_labels = candidate_probabilities.max(dim=1)

    zeta = min(zeta, available_idx.shape[0])
    top_confidence = torch.topk(confidence, k=zeta).indices

    selected_nodes = available_idx[top_confidence]
    selected_labels = class_labels[top_confidence]

    new_labels = labels.clone()
    new_train_mask = train_mask.clone()

    new_labels[selected_nodes] = selected_labels
    new_train_mask[selected_nodes] = True

    return new_labels, new_train_mask, selected_nodes, selected_labels