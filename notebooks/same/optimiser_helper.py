from finn.custom_op.registry import getCustomOp

import sys
sys.path.append("/workspace/same")

from optimiser.brute import optimise
from optimiser.node import Node

import networkx as nx

def optimiser_main(model):
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # create the computation graph
    graph = nx.DiGraph()
    ## add nodes
    for finn_node in fc_layers:
        graph.add_node(FinnNodeWrapper(getCustomOp(finn_node)))
    ## add edges
    for i in range(len(fc_layers)-1):
        graph.add_edge(list(graph.nodes)[i], list(graph.nodes)[i+1])

    # platform
    platform = {
        "LUT" : 53200,
        "DSP" : 220,
        "BRAM" : 280,
        "FF" : 106400
    }
    print("optimser start")
    # perform optimisation on the computation graph
    optimise(graph, platform)


class FinnNodeWrapper(Node):
    def __init__(self, finn_node):
        self.finn_node = finn_node

        self.channels_in = finn_node.get_nodeattr("MW")
        self.channels_out = finn_node.get_nodeattr("MH")

    def custom_update(self):
        self.finn_node.set_nodeattr("SIMD", self.channel_in_folding)
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


