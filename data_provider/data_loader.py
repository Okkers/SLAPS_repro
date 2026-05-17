### THIS IS BOILER PLATE FOR DATA LOADER CREATION
import os 
import numpy as np 
import torch
import pickle as pkl
import scipy.sparse as sp
import warnings 

warnings.filterwarnings('ignore')

def mask(idx, shape):
    """Creates a boolean mask for the given indices and shape.
    Args:
        idx (list): A list of indices to be masked.
        shape (tuple): The shape of the mask to be created.
    Returns:
        np.ndarray: A boolean mask with True at the specified indices and False elsewhere.
    """
    mask = np.zeros(shape)
    mask[idx] = 1
    mask = np.array(mask, dtype=bool)
    return mask

def extract_pickled_data(dataset_name):
    """Loads the pickled data for the specified dataset name.
    Args:
        dataset_name (str): The name of the dataset to load.
    Returns:
        tuple: A tuple containing the loaded data (x, y, tx, ty, allx, ally, graph, test_idx).
    """
    with open(os.path.join('data', 'ind.{}.x'.format(dataset_name)), 'rb') as f:
        x = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.y'.format(dataset_name)), 'rb') as f:
        y = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.tx'.format(dataset_name)), 'rb') as f:
        tx = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.ty'.format(dataset_name)), 'rb') as f:
        ty = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.allx'.format(dataset_name)), 'rb') as f:
        allx = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.ally'.format(dataset_name)), 'rb') as f:
        ally = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.graph'.format(dataset_name)), 'rb') as f:
        graph = pkl.load(f, encoding='latin1')
    with open(os.path.join('data', 'ind.{}.test.index'.format(dataset_name)), 'r') as f:
        test_idx = [int(line.strip()) for line in f]
    return (x, y, tx, ty, allx, ally, graph, test_idx)

def load_citation_data(dataset_name):
    """Loads the citation dataset for the specified dataset name.
    Args:
        dataset_name (str): The name of the citation dataset to load (e.g., 'cora', 'citeseer', 'pubmed').
    Returns:
        tuple: A tuple containing the features, labels, train_mask, val_mask, test_mask, number of features, and number of classes.
    """
    
    x, y, tx, ty, allx, ally, graph, test_idx = extract_pickled_data(dataset_name)

    # Citeseer dataset contains isolated nodes in the graph, so we find these and add them 
    # as zero-vecs into the missing positions in the tx and ty matrices. 
    if dataset_name == 'citeseer':
        test_idx_full_range = range(min(test_idx), max(test_idx)+1)
        sorted_tx = sp.lil_matrix((len(test_idx_full_range), x.shape[1]))
        sorted_ty = np.zeros((len(test_idx_full_range), y.shape[1]))

        sorted_tx[np.sort(test_idx) - min(test_idx)] = tx[np.argsort(test_idx)]
        sorted_ty[np.sort(test_idx) - min(test_idx)] = ty[np.argsort(test_idx)]
    else: 
        # For other datasets, we can directly sort the tx and ty matrices using the sorted test indices.
        test_idx = np.array(test_idx)
        test_idx_norm = np.argsort(test_idx)
        sorted_tx = tx[test_idx_norm]
        sorted_ty = ty[test_idx_norm]

    # Now we can concatenate the training and test data to create the full dataset.
    full_x = sp.vstack([allx, sorted_tx]).tolil()
    full_y = np.vstack([ally, sorted_ty])

    train_idx_values = range(len(y)) # since y contains all the training labels, we can use its length to determine the training indices.
    val_idx_values = range(len(y), len(y)+500) # Franceschi paper uses 500 nodes for validation
    test_idx_values = np.sort(test_idx).tolist() # contains 1000 nodes for testing

    mask_train = mask(train_idx_values, full_y.shape[0])
    mask_val = mask(val_idx_values, full_y.shape[0])
    mask_test = mask(test_idx_values, full_y.shape[0])

    features = torch.FloatTensor(full_x.todense())
    labels = torch.LongTensor(full_y)
    train_mask = torch.BoolTensor(mask_train)
    val_mask = torch.BoolTensor(mask_val)
    test_mask = torch.BoolTensor(mask_test)

    f = features.shape[1] # number of features

    # As we saw, some labels in citeseer are missing, so we need to account for this by doing the following:
    for i in range(labels.shape[0]):
        if labels[i].sum() == 0:
            labels[i] = torch.zeros(labels.shape[1])
            labels[i][0] = 1
    
    labels = torch.argmax(labels, dim=1)
    c = labels.max().item() + 1 # number of classes (0 is also a class)
    return features, labels, train_mask, val_mask, test_mask, f, c

# features, labels, train_mask, val_mask, test_mask, f, c = load_citation_data('citeseer')
    
def load_ogbn_arxiv_data():
    """Loads the OGBN-Arxiv dataset.
    Returns:
        tuple: A tuple containing the features, labels, train_mask, val_mask, test_mask, number of features, and number of classes.
    """

    # First load the datasets
    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    data = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
    dataset = data[0] # Data(num_nodes=169343, edge_index=[2, 1166243], x=[169343, 128], node_year=[169343, 1], y=[169343, 1])

    # Extract the features, labels, and masks from the dataset
    features = dataset.x
    labels = dataset.y
    f = features.shape[1]
    c = data.num_classes

    split_idx = data.get_idx_split()

    mask_train = mask(split_idx["train"], data.x.shape[0])
    mask_val = mask(split_idx["valid"], data.x.shape[0])
    mask_test = mask(split_idx["test"], data.x.shape[0])
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).squeeze() # shape (num_nodes,)
    train_mask = torch.BoolTensor(mask_train)
    val_mask = torch.BoolTensor(mask_val)
    test_mask = torch.BoolTensor(mask_test)
    
    return features, labels, train_mask, val_mask, test_mask, f, c

# load_ogbn_arxiv_data()

def mask_sklearn_data(features, labels, train_size, val_size, seed=42):
    """Creates train, validation, and test masks for a given dataset.
    Args:
        features (np.ndarray): The feature matrix of the dataset.
        labels (np.ndarray): The label vector of the dataset.
        train_size (int): The number of samples to include in the training set.
        val_size (int): The number of samples to include in the validation set.
        seed (int): The random seed for reproducibility.
    Returns:
        tuple: A tuple containing the train_mask, val_mask, and test_mask as boolean numpy arrays.
    """
    from sklearn.model_selection import train_test_split
    train_idx, temp_idx = train_test_split(
        np.arange(features.shape[0]),
        train_size=train_size,
        random_state=seed,
        stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=seed,
        stratify=labels[temp_idx]
    )

    train_mask = mask(train_idx, labels.shape[0])
    val_mask = mask(val_idx, labels.shape[0])
    test_mask = mask(test_idx, labels.shape[0])
    return train_mask, val_mask, test_mask

def load_sklearn_data(dataset_name):
    """Loads a dataset from scikit-learn.
    Args:
        dataset_name (str): The name of the dataset to load (e.g., 'wine', 'cancer', 'digits', '20news').
    Returns:
        tuple: A tuple containing the features, labels, train_mask, val_mask, test_mask, number of features, and number of classes.
    """
    # First we load the dataset using the appropriate scikit-learn function based on the dataset name.
    from sklearn.datasets import load_wine, load_breast_cancer, load_digits, fetch_20newsgroups_vectorized
    if dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "cancer":
        data = load_breast_cancer()
    elif dataset_name == "digits":
        data = load_digits()
    elif dataset_name == "20news":
        data = fetch_20newsgroups_vectorized(subset='all')
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    
    # After loading the dataset, extract the relevant information
    features = data.data
    labels = data.target
    f = features.shape[1]
    c = len(np.unique(labels))

    # Create train, val and test splits (predefined in the Franceschi paper)
    from sklearn.model_selection import train_test_split

    if dataset_name == "wine":
        train_mask, val_mask, test_mask = mask_sklearn_data(features, labels, train_size=10, val_size=20, seed=42)
    elif dataset_name == "cancer":
        train_mask, val_mask, test_mask = mask_sklearn_data(features, labels, train_size=10, val_size=20, seed=42)
    elif dataset_name == "digits":
        train_mask, val_mask, test_mask = mask_sklearn_data(features, labels, train_size=50, val_size=100, seed=42)
    elif dataset_name == "20news":
        train_mask, val_mask, test_mask = mask_sklearn_data(features, labels, train_size=100, val_size=200, seed=42)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    return features, labels, train_mask, val_mask, test_mask, f, c
    
# load_sklearn_data("wine")

def load_mnist_data(trainingset_size):

    """Loads the MNIST dataset and creates train, validation, and test masks.
    Args:
        labels_size (int): The number of samples to include in the training set (1000, 2000 or 3000).
    Returns:
        tuple: A tuple containing the features, labels, train_mask, val_mask, test_mask, number of features, and number of classes.
    """
    # First we load the MNIST dataset using torchvision.datasets.MNIST
    from torchvision.datasets import MNIST

    data = MNIST(root='data', train=True, download=True)

    features = data.data.view(-1, 28*28).float() / 255.0 # Normalize pixel values to [0, 1]
    labels = data.targets
    f = features.shape[1]
    c = 10 # digits 0-9

    # We need to extract 1000 random samples from each of the 10 classes (digits 0-9) to create a training set of 10,000 samples
    all_idx = []
    for digit in range(10):
        digit_idx = torch.where(labels == digit)[0]
        subset_idx = digit_idx[torch.randperm(len(digit_idx))[:1000]]
        all_idx.append(subset_idx)
    
    all_idx = torch.cat(all_idx)
    all_idx = all_idx[torch.randperm(len(all_idx))]

    features = features[all_idx]
    labels = labels[all_idx]
    
    # Create train, val and test splits 
    from sklearn.model_selection import train_test_split

    train_idx, temp_idx = train_test_split(
        np.arange(features.shape[0]),
        train_size=trainingset_size,
        random_state=42,
        stratify=labels.numpy()
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=1000,
        random_state=42,
        stratify=labels[temp_idx].numpy()
    )

    train_mask = mask(train_idx, labels.shape[0])
    val_mask = mask(val_idx, labels.shape[0])
    test_mask = mask(test_idx, labels.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    
    return features, labels, train_mask, val_mask, test_mask, f, c

# load_mnist_data(3000)