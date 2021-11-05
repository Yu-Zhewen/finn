from finn.custom_op.registry import getCustomOp
from finn.util.basic import get_by_name

import sys
sys.path.append("/workspace/same")

from optimiser.brute import BruteForce
from optimiser.annealing import SimulatedAnnealing
from optimiser.node import Node
from optimiser.optimiser import load_from_opt_network

from networkx import DiGraph

def parse(model):
    # create the computation graph
    network = DiGraph()
    ## add nodes
    for finn_node in model.graph.node:
        channels_in  = model.get_tensor_shape(finn_node.input[0])[-1]
        channels_out = model.get_tensor_shape(finn_node.output[0])[-1]
        network.add_node(finn_node.name, hw=FinnNodeWrapper(getCustomOp(finn_node), channels_in, channels_out))
    ## add edges
    for i in range(len(list(network.nodes))-1):
        network.add_edge(list(network.nodes)[i], list(network.nodes)[i+1])

    assert network.validate()

    return network

def optimiser_main(model):

    network = parse(model)

    # platform
    platform = {
        "LUT" : 53200,
        "DSP" : 220,
        "BRAM" : 280,
        "FF" : 106400
    }
    print("optimser start")
    # perform optimisation on the computation graph
    #opt = BruteForce(network, platform)
    #opt.optimise()

    opt = SimulatedAnnealing(network, platform)
    opt.optimise()
    load_from_opt_network(network, opt.network)

class FinnNodeWrapper(Node):
    def __init__(self, finn_node, channels_in, channels_out):
        self.finn_node = finn_node
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.constraints = { "matching_folding": self.finn_node.onnx_node.op_type not in ["StreamingFCLayer_Batch"] }

    def update(self):
        if get_by_name(self.finn_node.onnx_node.attribute, "SIMD") is not None:
            self.finn_node.set_nodeattr("SIMD", self.channel_in_folding)

        if get_by_name(self.finn_node.onnx_node.attribute, "PE") is not None:
            self.finn_node.set_nodeattr("PE", self.channel_out_folding)

    def latency_in(self, eval=False):
        return self.finn_node.get_exp_cycles()

    def latency_out(self, eval=False):
        return self.finn_node.get_exp_cycles()

    def resource(self):
        return {
             "LUT" : self.finn_node.lut_estimation(),
             "DSP" : self.finn_node.dsp_estimation(),
             "BRAM" : self.finn_node.bram_estimation(),
             "FF" : 0
        }


