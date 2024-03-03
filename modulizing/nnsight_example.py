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
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.to("cuda:0")

attn = model.transformer.h[3].attn

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

setattr(attn, wrapper_name, wrapper_module)
print(attn)

# %%

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):

    if wrapper_name not in gm._modules:
        gm.add_submodule(wrapper_name, wrapper_module)

    for node in gm.graph.nodes:    

        if node.op == 'call_method' and node.name == "tensor":

            with gm.graph.inserting_after(node):
                wrapper_args = (node.args[0], )
                wrapper_kwargs = node.kwargs
                wrapper_node = gm.graph.call_module(wrapper_name, args=wrapper_args, kwargs=wrapper_kwargs)
                node = wrapper_node
            
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

opt_model = torch.compile(attn, backend=custom_backend, dynamic=True, fullgraph=True)
gm = opt_model(attn_envoy._fake_inputs[0][0][0])
# %%

model.transformer.h[3].attn = opt_model
model

# %%

import Envoy
import importlib

importlib.reload(Envoy)

with nn_model.trace("empty"):
    real_input = nn_model.transformer.h[3].attn.input.save()

real_input = real_input.value
# %%

opt_model(real_input[0][0])
