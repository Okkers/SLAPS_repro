import torch.nn as nn
import torch
import torch.nn.functional as F 
import utils.graph_utils as graph_utils


class FullParametrization(nn.Module):
    def __init__(self, configs):
        super(FullParametrization, self).__init__()

        self.k = configs.gen_k

        # Initialize the adjacency matrix A as a learnable parameter using the kNN graph
        self.A = nn.Parameter(
            torch.tensor(graph_utils.initialize_kNN_graph(configs.features, configs.gen_k), dtype=torch.float32)
        )

    # Adding x just to keep consistency across generators
    def forward(self, x):
        return self.A
    
class MLP(nn.Module):
    def __init__(self, configs):
    # def __init__(self, layers_size, input_dim, k):
        super(MLP, self).__init__()


        # Note that the paper mentions that "we keep the input dimension the same throughout the layers."
        self.input_dim = configs.gen_input_dim
        self.output_dim = configs.gen_input_dim # the paper mentions that "for both variants, we initialize the weight matrices with identity"
                                        # so the output dimension is probably the same as the input dimension.
        self.k = configs.gen_k
        
        self.mlp = []

        if configs.gen_layers_size == 1:
            self.mlp.append(nn.Linear(configs.gen_input_dim, configs.gen_input_dim))
        else:
            for _ in range(configs.gen_layers_size - 1):
                self.mlp.append(nn.Linear(configs.gen_input_dim, configs.gen_input_dim))
                self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(configs.gen_input_dim, configs.gen_input_dim))

        self.mlp = nn.Sequential(*self.mlp)

        self.initialize_MLP()

    def forward(self, x):
        # Pass the input through the MLP layers
        x = self.mlp(x) 
        
        # We need kNN(MLP(X)), so we need to do the kNN operation
        A_tilde = graph_utils.kNN_generator(x, self.k)

        return A_tilde
    
    def initialize_MLP(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight.copy_(torch.eye(self.input_dim))
                    layer.bias.zero_()

class MLP_D(nn.Module):
    def __init__(self, configs):
        super(MLP_D, self).__init__()
        self.input_dim = configs.gen_input_dim
        self.output_dim = configs.gen_input_dim
        self.k = configs.gen_k

        self.mlp = nn.ParameterList()

        for _ in range(configs.gen_layers_size):
            self.mlp.append(nn.Parameter(torch.ones(configs.gen_input_dim)))


    def forward(self, x):
        # Pass the input through the MLP layers
        for idx, weight in enumerate(self.mlp):
            x = x * weight # since we have a diagonal matrix, we can just do element multiplication

            if idx != len(self.mlp) - 1:
                x = F.relu(x)
        
        # We need kNN(MLP(X)), so we need to do the kNN operation
        A_tilde = graph_utils.kNN_generator(x, self.k)

        return A_tilde