### THIS IS BOILER PLATE FOR DATA LOADING AND CREATION
from data_provider.data_loader import load_citation_data, load_ogbn_arxiv_data, load_sklearn_data, load_mnist_data
from torch_geometric.utils import negative_sampling
import torch

data_dict = {

}

def data_provider(args):
    """Returns the data for the specified dataset name.
    Args:
        args: An object containing the dataset name and other relevant parameters.
    Returns:
        The data for the specified dataset name, which include features, labels, train_mask, val_mask, test_mask, graph, f, c.
    """
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'cora390', 'citeseer370']:
        return load_citation_data(args.dataset)
    elif args.dataset == "ogbn-arxiv":
        return load_ogbn_arxiv_data()
    elif args.dataset == "mnist":
        return load_mnist_data(args.mnist_training_size)
    elif args.dataset in ["wine", "cancer", "digits", "20news"]:
        return load_sklearn_data(args.dataset)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")



def apply_noise_to_graph(edge_index, num_nodes, rho):
    """
    Function used to recreate figure 5
    """
    # percentage!
    rho = rho * 1 / 100

    if rho == 0.0:
        return edge_index.clone()

    num_edges = edge_index.size(1)
    num_to_replace = int(num_edges * rho)

    # randomly select edges to remove
    perm = torch.randperm(num_edges)
    keep_indices = perm[num_to_replace:]
    kept_edges = edge_index[:, keep_indices]

    # generate new random edges
    new_random_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_to_replace,
    )

    # combine all edges together
    perturbed_edge_index = torch.cat([kept_edges, new_random_edges], dim=1)

    return perturbed_edge_index