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

from torch._guards import detect_fake_mode

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):

    if wrapper_name not in gm._modules:
        gm.add_submodule(wrapper_name, wrapper_module)

    for node in gm.graph.nodes:    

        if node.op == 'call_method' and node.name == "tensor":
            if node.args[0].name == "query":
                print('found')
                with gm.graph.inserting_before(node):
                    wrapper_args = (node.args[0], )
                    wrapper_node = gm.graph.call_module(wrapper_name, args=wrapper_args)
                    
                    node.update_arg(0, wrapper_node)
            
    gm.recompile()
    print(gm)

    # gm.graph.print_tabular()

    return gm.forward

torch._dynamo.reset()

# torch._dynamo.allow_in_graph(detect_fake_mode)

opt_model = torch.compile(attn, backend=custom_backend, dynamic=True)
# opt_model = torch.compile(attn, backend=custom_backend, dynamic=True, fullgraph=True)
gm = opt_model(attn_envoy._fake_inputs[0][0][0])

# %%

model.transformer.h[3].attn = opt_model
model

# %%

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
new_model = LanguageModel(model, tokenizer=tokenizer)

# %%
with new_model.trace("Please work", scan=False):
    test_save = new_model.transformer.h[3].attn._orig_mod.output_wrapper.output.save()
