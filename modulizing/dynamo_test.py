# %%
from typing import List

import torch
import torch.nn as nn

from nnsight import LanguageModel
from nnsight.util import WrapperModule

from nnsight import NNsight
# %%

class WrappedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = x * 100
        return x

class WrappedLayerTwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)

    # @torch.compiler.disable()
    def forward(self, x):
        x = self.layer1(x)

        x = torch.mul(x, 5)
        return x

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)
        self.wrapped_one = WrappedLayer()
        self.wrapped_two = WrappedLayerTwo()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.wrapped_one(x)
        x = self.wrapped_two(x)
        x = self.dropout(x)
        x = x.split(1, dim=-1)
        return x

# Example usage
mod = M()

# Assuming the input to be of shape (batch_size, num_features), suitable for both Linear and BatchNorm1d layers
input_tensor = torch.tensor([[1.0]])  # Shape: (batch_size=1, num_features=1)
output = mod(input_tensor)

print(mod)
print(output)
# %%



#%%
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

# setattr(mod.wrapped_two.forward, "_torchdynamo_disable", True)
torch._dynamo.decorators.skip(WrappedLayer.forward)
opt_model = torch.compile(mod, backend=custom_backend, dynamic=True)

gm = opt_model(torch.tensor([1.0]))

# %%

nn_mod = NNsight(mod)
# %%

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

torch.compiler.allow_in_graph(torch.nn.modules.module.register_module_backward_hook)
opt_model = torch.compile(nn_mod._model, backend=custom_backend, dynamic=True)
# torch.compiler.disable(torch.split)
gm = opt_model(torch.tensor([1.0]))


# %%


from nnsight import NNsight
nn_mod = NNsight(mod)

# %%
nn_mod._envoy.clear_hooks()

# %%

print(nn_mod._model.__dict__)
nn_mod._envoy._hook_handle.remove()
nn_mod._model.__dict__

# %%



def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

opt_model = torch.compile(nn_mod._model, backend=custom_backend, dynamic=True, fullgraph=True)
gm = opt_model(torch.tensor([1.0]))

# %%

torch._dynamo.reset()
explain_output = torch._dynamo.explain(mod)(torch.tensor([1.0]))
print(explain_output)



# %%

from nnsight.envoy import Envoy

nn_model._model.transformer.h[3].attn = opt_model

nn_model._envoy = Envoy(nn_model._model)

# %%
with nn_model.trace("Please work", scan=False):
    test_save = nn_model.transformer.h[3].attn._orig_mod.output_wrapper.output.save()

# %%

nn_model._model.transformer.h[3].attn = nn_model._model.transformer.h[3].attn._orig_mod

nn_model._envoy = Envoy(nn_model._model)
# %%
with nn_model.trace("Please work", scan=False):
    test_save = nn_model.transformer.h[3].attn._orig_mod.output_wrapper.output.save()
# %%

test_save
# %%
