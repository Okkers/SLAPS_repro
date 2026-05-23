import torch
import torch.nn.functional as F


class AdjacencyProcessor:
    """
    Depending on generator type we apply different function P, either relu or elu.
    """

    def __init__(self, generator_type):
        self.generator_type = generator_type

    def function_p(self, A_tilde):
        if self.generator_type == 'fp':
            return F.elu(A_tilde) + 1
        else:
            return F.relu(A_tilde)


    def apply_adj_processor(self, A_tilde):
        """
        Applies the adjacency processor to A_tilde.
        :param A_tilde: Output of the generator
        :return: normalized A_tilde
        """

        P_A_tilde = self.function_p(A_tilde)

        # Symmetrize
        A_sym = 0.5 * (P_A_tilde + P_A_tilde.T)

        # Degree matrix D_inv
        degrees = A_sym.sum(dim=1)
        D_inv_sqrt = torch.where(degrees > 0, degrees.pow(-0.5), 0.0) # to avoid division by 0

        A = D_inv_sqrt.unsqueeze(1) * A_sym * D_inv_sqrt.unsqueeze(0)

        return A
