from __future__ import annotations

from typing import Any, Mapping

from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import Parameter, ParameterFactory
from cirkit.symbolic.parameters import TensorParameter
from cirkit.symbolic.initializers import NormalInitializer, ConstantTensorInitializer
from cirkit.backend.torch.layers.inner import TorchSumLayer
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
from cirkit.backend.torch.rules.layers import DEFAULT_LAYER_COMPILATION_RULES
from nystromlayer import NystromSumLayer
import torch


def compile_nystrom_sum_layer(compiler, sl: "NystromSumLayer") -> NystromSumLayer:
    """Compile the symbolic layer to a torch ``NystromSumLayer``."""
    U = compiler.compile_parameter(sl.U)
    V = compiler.compile_parameter(sl.V)

    U_val = U()
    V_val = V()
    weight_val = torch.einsum("fok,fik->foi", U_val, V_val)
    init = compiler.compile_initializer(
        ConstantTensorInitializer(weight_val.detach().cpu().numpy())
    )
    weight = TorchTensorParameter(
        sl.num_output_units,
        sl.num_input_units * sl.arity,
        requires_grad=False,
        initializer_=init,
        num_folds=U.num_folds,
    )
    dense_layer = TorchSumLayer(
        sl.num_input_units,
        sl.num_output_units,
        arity=sl.arity,
        weight=weight,
        semiring=compiler.semiring,
    )
    nys = NystromSumLayer(dense_layer, rank=sl.rank)
    nys.weight_orig = weight_val.detach()
    return nys




class NystromSumLayer(Layer):
    """Symbolic Nyström-compressed sum layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        rank: int,
        arity: int = 1,
        *,
        U: Parameter | None = None,
        V: Parameter | None = None,
        U_factory: ParameterFactory | None = None,
        V_factory: ParameterFactory | None = None,
    ) -> None:
        super().__init__(num_input_units, num_output_units, arity=arity)
        self.rank = int(rank)

        if U is None:
            if U_factory is None:
                U = Parameter.from_input(
                    TensorParameter(num_output_units, rank, initializer=NormalInitializer())
                )
            else:
                U = U_factory((num_output_units, rank))

        if V is None:
            if V_factory is None:
                V = Parameter.from_input(
                    TensorParameter(num_input_units, rank, initializer=NormalInitializer())
                )
            else:
                V = V_factory((num_input_units, rank))

        self.U = U
        self.V = V

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "rank": self.rank,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"U": self.U, "V": self.V}


DEFAULT_LAYER_COMPILATION_RULES[NystromSumLayer] = compile_nystrom_sum_layer

