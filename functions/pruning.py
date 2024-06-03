import torch
from torch.nn.utils.prune import BasePruningMethod

class EveryOtherPruning(BasePruningMethod):
    """
    Prunes every other entry in a tensor.
    """

    PRUNING_TYPE = 'structured'

    def __init__(self, amount):
        """
        Initialize the pruning method.

        Args:
            amount (float): The percentage of connections to prune.
        """
        super(EveryOtherPruning, self).__init__()
        self.amount = amount

    def compute_mask(self, tensor, default_mask):
        """
        Compute the mask to prune every other entry in the tensor.

        Args:
            tensor (torch.Tensor): The tensor to prune.
            default_mask (torch.Tensor): The default mask provided by the BasePruningMethod.

        Returns:
            torch.Tensor: The mask indicating which entries to prune.
        """
        if self.amount == 1:
            return torch.zeros_like(tensor, dtype=torch.bool)
        else:
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[::2] = 0  # Keep every other entry unpruned
            return mask

# Example usage:
# Create a module
module = torch.nn.Linear(10, 10)

# Apply pruning
prune.ln_structured(module, name="weight", amount=0.5, n=1, dim=0)