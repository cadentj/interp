import torch as t 

class Nneuron(t.nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """

    def __init__(self,param_shape=None):
        super().__init__()
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name = None
        self.param_shape = param_shape  # Store the parameter shape information
        self.last_input_shape = None
        
    def forward(self, x):
        self.last_input_shape = tuple(x.shape)
        return x

    def set_param_info(self, param_shapes):
        self.param_shapes = param_shapes

    def __repr__(self):
        
        shape_info = f"param_shape={self.param_shape}" if self.last_input_shape == None else f"param_shape={self.last_input_shape}"

        # shape_info = f"param_shape={self.param_shape}" if self.param_shape else "param_shape=unknown"
        return f"{self.__class__.__name__}({shape_info})"