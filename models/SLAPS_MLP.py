import torch 
import torch.nn as nn 
from layers.generator import MLP
from layers.GNN_C import GNN_C
from layers.GNN_DAE import GNN_DAE
from layers.adjacency_processor import AdjacencyProcessor

class Model(nn.Module):
    '''
    Model that absolutely SLAPS.
    Reproduction and code constructed from:
    SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks, (2021)
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_dim_dae = configs.hidden_dim_dae
        self.out_dim = configs.output_dim
        self.is_discrete = configs.is_discrete
        self.r = configs.r
        self.eta = configs.eta
        self.noise_type = configs.noise_type


        self.generator = MLP(configs)

        # SELF.ADJACENCY_PROCESSOR NEEDED
        self.adjacency_processor = AdjacencyProcessor()

        # GCNs ALSO NEED IMPLEMENTATION
        self.gcn_c = GNN_C(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.out_dim,
            dropout_c=configs.dropout_c
        )

        self.gcn_dae = GNN_DAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim_dae,
            output_dim=self.input_dim,
            dropout_adj=configs.dropout_DAE
        )

        # self.lambda_weight = configs.lambda_weight


    def add_noise(self, x):
        # CAREFUL HERE, THIS DOES NOT APPLY IF WE CAN ENCOUNTER THE SAME SAMPLE MULTIPLE TIMES PER EPOCH

        x_tilde = x.clone()
        n, f = x.shape 


        # If dataset is discrete, (See explanation under Equation (1) in paper )
        if self.is_discrete:
            ones_mask = x == 1 
            zeros_mask = x == 0 

            # Let there be r% of ones corrupted (note. r should be passed as r=0.25 for %)
            num_ones = ones_mask.sum().item()
            nums_ones_to_noise = int(round((self.r / 100) * num_ones))
            ones_indices = ones_mask.view(-1).nonzero(as_tuple=False).squeeze()
            the_CHOSEN_ONES = ones_indices[torch.randperm(len(ones_indices))[:nums_ones_to_noise]]

            # Let there be r*eta% of zeros corrupted
            num_zeros = zeros_mask.sum().item()
            num_zeros_to_noise = int(round(self.eta * nums_ones_to_noise))
            zeros_indices = zeros_mask.view(-1).nonzero(as_tuple=False).squeeze()
            the_CHOSEN_zeros = zeros_indices[torch.randperm(len(zeros_indices))[:num_zeros_to_noise]]

            # X_idx
            all_indices = torch.cat([the_CHOSEN_ONES, the_CHOSEN_zeros], dim=0)
            mask = torch.zeros_like(x, dtype=torch.bool).view(-1)
            mask[all_indices] = True 
            mask = mask.view(n, f)

            x_tilde[mask] = 0.0
        
        # If dataset is continuous (Same section as before)
        else:
            num_to_noise = int(round((self.r / 100) * n * f))  # n*f = x.dim1 * x.dim2
            the_chosen_indices = torch.randperm(n * f, device=x.device)[:num_to_noise]

            mask = torch.zeros(n * f, dtype=torch.bool, device=x.device)
            mask[the_chosen_indices] = True
            mask = mask.view(n, f)

            if self.noise_type == "zero":
                x_tilde[mask] = 0.0 
            
            # If not zero -> gaussian
            else:
                noise = torch.randn_like(x)
                x_tilde[mask] = x_tilde[mask] + noise[mask]


        # We need mask for X_idx later, so we return it 
        return x_tilde, mask


    def forward(self, x):
        adj = self.adjacency_processor.apply_adj_processor(self.generator(x))

        x_tilde, mask = self.add_noise(x)
    
        logits_c = self.gcn_c(adj, x)

        x_reconstructed = self.gcn_dae(adj, x_tilde)

        # Still need mask for the loss -> return it
        return logits_c, x_reconstructed, mask
