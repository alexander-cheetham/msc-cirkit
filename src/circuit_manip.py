from types import MethodType
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer

# Legacy utilities ``replace_sum_layers`` and ``fix_address_book_modules`` have
# been moved to :mod:`legacy.circuit_manip`.

def build_circuit_one_sum(
    self,
    *,
    input_factory,        # like lambda scope, n: GaussianLayer(scope, n)
    sum_weight_factory,   # your ParameterFactory for the SumLayer
    num_input_units: int, # # of outputs per Gaussian leaf
    num_sum_units: int,# # of mixtures in the one SumLayer
    debug=False,
) -> Circuit:
    # 1) Find all the leaves in the region graph:
    leaves = [node for node in self.topological_ordering()
              if not self.region_inputs(node)]
    
    layers = []
    in_layers = {}
    
    # 2) Build one GaussianLayer per leaf
    gaussians = []
    for leaf in leaves:
        gauss = input_factory(leaf.scope, num_input_units)
        layers.append(gauss)
        gaussians.append(gauss)
    
    # 3) Build *one* SumLayer mixing them all
    sum_layer = SumLayer(
        num_input_units=num_input_units,
        num_output_units=num_sum_units,
        arity=len(gaussians),
        weight_factory=sum_weight_factory,
    )
    layers.append(sum_layer)
    in_layers[sum_layer] = gaussians
    
    # 4) Return a circuit whose only output is that top‐sum
    if debug:
        print(layers,"---------------\n\n\n",in_layers,"---------------\n\n\n",[sum_layer])
    return Circuit(layers, in_layers, outputs=[sum_layer])



import cirkit.symbolic.functional as SF
from cirkit.symbolic.io import plot_circuit
from cirkit.pipeline import PipelineContext
from helpers import define_circuit_one_sum

def build_and_compile_circuit(input_units: int, sum_units: int, *, nystrom: bool = False):
    """
    Build a one‐sum circuit with the given number of input and sum units,
    compile it in a sum‐product semiring, and return only the compiled circuit.
    All intermediate variables are deleted before returning.
    """
    # Build symbolic network
    net = define_circuit_one_sum(input_units, sum_units)

    # Compile network to a torch backend
    ctx = PipelineContext(
        backend="torch",
        semiring="sum-product",
        fold=False,
        optimize=False,
        nystrom=nystrom,
    )
    cc = ctx.compile(net).cpu().eval()

    # Build and plot the partition‐function circuit
    symbolic_circuit_partition_func = SF.multiply(net, net)
    plot_circuit(symbolic_circuit_partition_func)

    # Compile the partition function circuit
    csc = ctx.compile(symbolic_circuit_partition_func)

    # (Optional) inspect a particular parameter
    kronparameter = csc.layers[1].weight._nodes[4]

    # Delete everything except the result
    del net, cc, ctx, symbolic_circuit_partition_func, kronparameter
    return csc
