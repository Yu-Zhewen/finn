import sys
sys.path.append("/workspace/same")

from backend.finn.parser import parse

from optimiser.brute import BruteForce
from optimiser.annealing import SimulatedAnnealing
from optimiser.optimiser import load_from_opt_network

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






