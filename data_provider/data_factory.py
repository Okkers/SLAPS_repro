### THIS IS BOILER PLATE FOR DATA LOADING AND CREATION
from data_provider.data_loader import load_citation_data, load_ogbn_arxiv_data, load_sklearn_data, load_mnist_data

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
