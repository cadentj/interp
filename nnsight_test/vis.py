# %%
from __future__ import annotations

from nnsight.patching import Patch, Patcher
from nnsight.tracing.Node import Node
from nnsight.tracing.Proxy import Proxy, proxy_wrapper


import inspect
from typing import Any, Callable, Dict, List, Type, Union

import torch

from nnsight import LanguageModel
from nnsight.tracing.Graph import Graph
from nnsight.module import Module
from nnsight.util import apply


# %%

model = LanguageModel("gpt2", device_map="auto", dispatch=True)

a = apply(
    model.transformer.input_shape,
    lambda x: torch.empty(x, device="meta"),
    torch.Size,
)




# %%

@staticmethod
def trace(
    module: torch.nn.Module, *args: List[Any], **kwargs: Dict[str, Any]
) -> Graph:
    """Given a module and some default (should be meta tensors) arguments, create a graph from the module's
    forward method.

    Args:
        module (torch.nn.Module): _description_
        args (List[Any]): desc
        kwargs (Dict[str, Any]): desc

    Returns:
        Graph: _description_
    """

    # Create a graph with the module as the root module
    graph = Graph(module)

    # Get 'unbound' version of forward method so we can pass in proxy of module instead of self
    forward = module.__class__.forward

    # Want list not tuple
    args = list(args)

    # Inspect forward signature to collect all parameters
    signature = inspect.signature(forward)

    def get_argument_value(param: inspect.Parameter, idx: int):
        """Gets the correct argument to pass to forward method.

        Args:
            param (_type_): _description_
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """

        # If idx in range of provided args, create a proxy for that arg instead of default.
        if idx < len(args):
            return graph.add(
                value=args[idx], target="argument", args=[param.name]
            )
        # If param name in provided kwargs, create a proxy for that arg instead of default.
        if param.name in kwargs:
            return graph.add(
                value=kwargs[param.name],
                target="argument",
                args=[param.name],
            )
        # Otherwise just return default

        return param.default

    # Create the appropriate proxies/values for the forward method in order to trace.
    arguments = [
        get_argument_value(param, i)
        for i, param in enumerate(list(signature.parameters.values())[1:])
    ]

    # Some methods cannot be caught because they aren't torch functions or dont play nice with __torch_function__.
    # So the patcher replaces the methods with something to catch proxies and return proxies.
    with Patcher() as patcher:
        patcher.add(Patch(torch, proxy_wrapper(torch.full), "full"))
        patcher.add(Patch(torch, proxy_wrapper(torch.finfo), "finfo"))
        patcher.add(Patch(torch, proxy_wrapper(torch.arange), "arange"))

        # Run forward with root module proxy and arguments
        output: Proxy = forward(graph.module_proxy, *arguments)

        # Create the 'swap' return proxy
        return_proxy = graph.add(
            graph=graph, value=True, target="swp", args=[output.node, output.node]
        )

    return graph


val = trace(model.transformer, *a)



# %%
