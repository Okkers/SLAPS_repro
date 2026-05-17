import torch
import torch.nn.functional as F


class AdjacencyProcessor:
    """
    Depending on generator type we apply different function P, either relu or elu.
    """

    def __init__(self, generator_output, generator_type):
        self.A_tilde = generator_output
        self.generator_type = generator_type

    def function_p(self):
        if self.generator_type == 'fp':
            return self.apply_elu()
        else:
            return self.apply_relu()

    def apply_relu(self):
        return F.relu(self.A_tilde)

    def apply_elu(self):
        return F.elu(self.A_tilde) + 1

    def apply_adj_processor(self):

        P_A_tilde = self.function_p()

        # Symmetrize
        A_sym = 0.5 * (P_A_tilde + P_A_tilde.T)

        # Degree matrix D_inv
        degrees = A_sym.sum(dim=1)
        D_inv_sqrt = torch.where(degrees > 0, degrees.pow(-0.5), 0.0) # to avoid division by 0

        A = D_inv_sqrt.unsqueeze(1) * A_sym * D_inv_sqrt.unsqueeze(0)

        return A
