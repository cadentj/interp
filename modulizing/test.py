# %%

from nnsight import LanguageModel
from nnsight.edit import Edit
from nnsight.util import WrapperModule
import torch


# %%

model = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

# %%

class EditModule(torch.nn.Module):

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        value = args * 1000
        
        return value
    
edit = Edit(
    model._envoy.transformer.h[3].attn._module_path, 
    "value", 
    "value_wrapper",
    EditModule()
)

class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args
    
wrapper_edit = Edit(
    model._envoy.transformer.h[3].attn._module_path, 
    "query", 
    "query_wrapper",
    WrapperModule()
)


# edits = [edit, wrapper_edit]
edits = [wrapper_edit]

model.load_edits(edits)

# %%

with model.trace("empty", scan=False, validate=False):
    query = model.transformer._orig_mod.h[3].attn.query_wrapper.output.save()
    # value = model.transformer._orig_mod.h[3].attn.value_wrapper.output.save()

    # metric = model.lm_head.output.sum()

    # metric.backward()

print(query)
# print(value)
# %%
print(model)

# model.clean_edits()
# %%

from nnsight.edit import print_gm

with model.trace('test'):
    pass

# %%

model._envoy.transformer.h[3].attn.clear_hooks()
# %%
print_gm(model._envoy.transformer.h[3].attn)
# %%

