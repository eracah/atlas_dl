
import matplotlib; matplotlib.use("agg")


import sys
from scripts.configs.setup_configs import setup_configs
from scripts.trainer import TrainVal



if __name__ == "__main__":
    
    configs = setup_configs()
    networks, fns = configs["net"].build_network(configs, configs["net"].build_layers(configs))
    tv = TrainVal(configs, fns, networks)
    tv.train()

    
    
    









