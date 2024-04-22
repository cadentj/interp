from nnsight import LanguageModel
import torch
from typing import List

model = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

with model.trace("a"):
    h_0_input = model.transformer.h[0].input.save()
    attn_input = model.transformer.h[0].attn.input.save()
    
    value_to_comapre = model.transformer.h[0].attn.output.save()

class Edit:

    def __init__(self, 
        parent: str, 
        target: str, 
        key: str, 
        replacement: torch.nn.Module,
    ) -> None:
        self.parent = parent
        self.target = target
        self.key = key
        self.replacement = replacement

    def __str__(self) -> str:
        return f"{self.parent}.{self.target} -> {self.key}"
    
    def __repr__(self) -> str:
        return self.__str__()

class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        value = args * 10000000
    
        print("stuff is going through this node!")

        return value

edit = Edit(
    "transformer.h.0.attn", 
    "query", 
    "query_wrapper",
    WrapperModule()
)
    

def edited_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):

    gm.add_submodule(edit.key, edit.replacement)

    for node in gm.graph.nodes:
        
        if node.name == "query":
            with gm.graph.inserting_before(node):
                new = gm.graph.create_node(node.op, node.target, args=node.args, kwargs=node.kwargs, name="original_" + node.name)
                wrapper_node = gm.graph.call_module(edit.key, args=(new,))
                node.replace_all_uses_with(wrapper_node)
                gm.graph.erase_node(node)


    gm.recompile()

    gm.graph.print_tabular()

    return gm.forward

mod = model._model.transformer.h[0].attn

setattr(mod, edit.key, edit.replacement)

torch._dynamo.reset()

opt_model = torch.compile(model._model.transformer.h[0].attn, backend=edited_backend, dynamic=True)
gm = opt_model(attn_input.value[0][0])

model._model.transformer.h[0].attn = opt_model

from nnsight.envoy import Envoy
model._envoy = Envoy(model._model)

with model.trace("empty", scan=False, validate=False):
    model.transformer.h[0].attn._orig_mod.query_wrapper.output *= 100
    out_two = model.transformer.h[0].attn.output.save()

value_to_comapre == out_two