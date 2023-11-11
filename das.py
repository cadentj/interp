import torch
from abc import ABC, abstractmethod
from torch.nn import CrossEntropyLoss

# class Intervention(torch.nn.Module, ABC):

#     """Intervention the original representations."""
#     def __init__(self):
#         super().__init__()
#         self.trainble = False
        
#     @abstractmethod
#     def set_interchange_dim(self, interchange_dim):
#         pass

#     @abstractmethod
#     def forward(self, base, source):
#         pass
    
    
# class TrainbleIntervention(Intervention):

#     """Intervention the original representations."""
#     def __init__(self):
#         super().__init__()
#         self.trainble = True



# Boundless DAS Acc

# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = (actual_test_labels==pred_test_labels)
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count/total_count, 2)
    return {"accuracy" : accuracy}

# Boundless DAS Loss
vocab_size = 32001
def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    boundary_loss = 1. * rotatedSpaceIntervention.intervention_boundaries.sum()
    loss += boundary_loss
    
    return loss


def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate sigmoid mask"""
    return torch.sigmoid((_input - boundary_x) / temperature) * \
        torch.sigmoid((boundary_y - _input) / temperature)

class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        # we don't need init if the saved checkpoint has a nice
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class BoundlessRotatedSpaceIntervention(torch.nn.Module):
    
    """Intervention in the rotated space with boundary mask."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = RotateLayer(embed_dim)

        # Orthogonal parametrizations constrains the weights of the rotate layer to be 
        # orthogonal during training
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer)

        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(50.0)) 
        self.embed_dim = embed_dim
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False)
        
    def get_boundary_parameters(self):
        return self.intervention_boundaries

    # temporarily in here
    def zero_grad(self):
        """Zero out the gradients of all parameters in this module."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp
        
    def set_interchange_dim(self, interchange_dim):
        """interchange dim is learned and can not be set"""
        assert False

    def forward(self, base, source):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(
            self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1), 
            0.,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature
        )
        boundary_mask = torch.ones(
            batch_size, device=base.device).unsqueeze(dim=-1)*boundary_mask
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (1. - boundary_mask)*rotated_base + boundary_mask*rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention(embed_dim={self.embed_dim})"