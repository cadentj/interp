# %%
from typing import List

import torch

from nnsight import LanguageModel
from nnsight.util import WrapperModule

# %%
nn_model = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

_ = nn_model.trace("empty", trace=False)

nn_attn = nn_model._model.transformer.h[3].attn
attn_envoy = nn_model._envoy.transformer.h[3].attn

# %%

class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args
    
wrapper_module = WrapperModule()
wrapper_name = 'output_wrapper'

setattr(nn_attn, wrapper_name, wrapper_module)
print(nn_attn)

# %%

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):

    if wrapper_name not in gm._modules:
        gm.add_submodule(wrapper_name, wrapper_module)

    found = False
    for node in gm.graph.nodes:    
        arg_names = [arg.name for arg in node.args if hasattr(arg, "name")]
        if "query" in arg_names:
            query_index = arg_names.index("query")

            with gm.graph.inserting_after(node):
                wrapper_args = (node.args[query_index], )
                wrapper_kwargs = node.kwargs
                wrapper_node = gm.graph.call_module(wrapper_name, args=wrapper_args, kwargs=wrapper_kwargs)
                node = wrapper_node

            found = True

        if found:
            break
        

            

            
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

opt_model = torch.compile(nn_attn, backend=custom_backend, dynamic=True)
gm = opt_model(attn_envoy._fake_inputs[0][0][0])

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
