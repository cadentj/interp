# %% 

from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

with model.trace(" "):
    pass

# %%

from nnsight.util import fetch_attr
import torch

def clear_forward(model, target):

    module_list = list(dict(model.named_modules()).keys())

    components = target.split('.')
    h_components = []

    for i in range(1, len(components)):
        c = '.'.join(components[:i])
        h_components.append(c)
    h_components.append(target)

    filtered_module_list = []
    for m in module_list:
        if m in h_components:
            continue
        if components[-1] in m:
            continue 
        filtered_module_list.append(m)

    for m in filtered_module_list:
        module = fetch_attr(model, m)
        if hasattr(module, "forward"):
            torch._dynamo.decorators.skip(module.__class__.forward)

    

# %%

from typing import List
import time

def custom_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

opt_model = torch.compile(model._model.transformer, backend=custom_backend, dynamic=True)

start_time = time.time()
gm = opt_model(torch.tensor([1]).to("cuda:0"))
end_time = time.time()

print(end_time - start_time)

# %%

def custom_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

opt_model = torch.compile(model._model.transformer, backend=custom_backend, dynamic=True)

clear_forward(model._model, "transformer.h.0.attn.c_attn")
start_time = time.time()
gm = opt_model(torch.tensor([1]).to("cuda:0"))
end_time = time.time()

print(end_time - start_time)


# %%

# CHECK IF WE CAN OPT ALR OPT MODULE

from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

with model.trace("a"):
    h_0_input = model.transformer.h[0].input.save()
    attn_input = model.transformer.h[0].attn.input.save()

# %%


import torch
from typing import List

attn = model._model.transformer.h[0].attn

def custom_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

torch._dynamo.reset()

opt_model = torch.compile(attn, backend=custom_backend, dynamic=True)

gm = opt_model(attn_input.value[0][0])

model._model.transformer.h[0].attn = opt_model
h_0 = model._model.transformer.h[0]
# h_0_envoy = model._envoy.transformer.h[0]

print(h_0)

# %%

def custom_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward

# torch._dynamo.reset()

opt_model_two = torch.compile(h_0, backend=custom_backend, dynamic=True)

gm_two = opt_model_two(h_0_input.value[0][0])

# %%

print(opt_model_two)

# %%

from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

with model.trace("a"):
    # h_0_input = model.transformer.h[0].input.save()
    # attn_input = model.transformer.h[0].attn.input.save()
    pass

# %%


edits = [
    "model.transformer.h.0.attn",
    "model.transformer.h.0.attn.c_proj",
    "model.transformer.h.1.attn.c_attn",
    "model.transformer.h.0.attn",
    "model.transformer.h.0.attn",
    "model.transformer.h.0.attn",
    "model.transformer.h.0.attn.mlp.c_proj",
    "model.transformer.h.0.attn.mlp",
    "model.lm_head",
    "model.wte"
]

def group_edits(edits):
    grouped_edits = {}
    # Sort the edits to check parents before children
    sorted_edits = sorted(edits, key=lambda x: x.count('.'))
    
    for current_edit in sorted_edits:
        # Assume an edit is a parent if it is a substring of another edit, and it's not the same edit
        for potential_parent in sorted_edits:
            if potential_parent in current_edit and potential_parent != current_edit:
                # Check if the potential parent is already in the grouped_edits
                if potential_parent in grouped_edits:
                    grouped_edits[potential_parent].append(current_edit)
                else:
                    grouped_edits[potential_parent] = [current_edit]
                break 

    # Check all edits are in the grouped_edits
    for edit in edits:
        if edit not in grouped_edits and not any(edit in subedits for subedits in grouped_edits.values()):
            grouped_edits[edit] = []
    
    return grouped_edits

group_edits(edits)