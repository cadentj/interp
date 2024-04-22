# %%

import operator
import torch
import torch._dynamo.testing
from typing import List


class CompileCounterWithBackend:
    def __init__(self, backend, fw_compiler=None, bw_compiler=None):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.fw_compiler = fw_compiler
        self.bw_compiler = bw_compiler

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        from torch._dynamo.eval_frame import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        if self.backend == "aot_autograd":
            return lookup_backend(self.backend)(
                gm,
                example_inputs,
                fw_compiler=self.fw_compiler,
                bw_compiler=self.bw_compiler,
            )
        return lookup_backend(self.backend)(gm, example_inputs)


def trace_printer(gm, _):
    # print(f"{'*'*128}\nFX Graph as Readable:\n{gm.print_readable()}")
    print(f"{'*'*128}\n FX Graph as Tabular\n{'*'*128}")
    print(gm.graph.print_tabular())
    return gm


def test_allow_in_graph_with_operator_lt():
    class MyModule(torch.nn.Module):
        def forward(self, a):
            x = torch.add(a, 1)
            y = torch.add(x, 1)
            if x.sum() < 0:
                x += torch.add(x, 1)
                y += torch.add(x, 1)
            return x + y

    nopython = False
    data = torch.randn(10)
    model = MyModule()
    torch._dynamo.allow_in_graph(operator.lt)
    compile_counter = CompileCounterWithBackend(
        "aot_autograd", fw_compiler=trace_printer
    )
    dynamo_model = torch._dynamo.optimize(compile_counter, nopython=nopython)(model)
    dynamo_model(data)
    torch._dynamo.disallow_in_graph(operator.lt)

    assert (
        compile_counter.frame_count == 1
    ), f"{compile_counter.frame_count} graph breaks were found!"


test_allow_in_graph_with_operator_lt()
# %%
