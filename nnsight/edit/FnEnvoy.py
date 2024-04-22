from __future__ import annotations

from nnsight.envoy import Envoy
from typing import Callable, Union, Any
from nnsight.intervention import InterventionProxy

class FnEnvoy(Envoy):
    
    def __init__(
            self, 
            envoy: Envoy, 
            fn: Callable, 
            inverse: Callable, 
        ):
        super().__init__(envoy._module)

        self._envoy = envoy
        self._fn = fn
        self._inverse = inverse

        self._output = None

    def __repr__(self):
        
        return "Placeholder"

    @property
    def output(self):
        """
        Calling denotes the user wishes to get the output of the underlying module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            InterventionProxy: Output proxy.
        """
        if self._output is None:

            self._output = self._tracer._graph.add(
                target=self._fn,
                args=[
                    self._envoy.output,
                ],
            )

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the output of the underlying module and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """

        value = self._inverse(value)

        self._tracer._graph.add(
            target="swap", args=[self.output.node, value], value=True
        )

        self._output = None