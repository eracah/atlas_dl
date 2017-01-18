
import matplotlib; matplotlib.use("agg")


import sys
from scripts.setup_configs import setup_configs, update_configs
from scripts.train_val import TrainVal



if __name__ == "__main__":
    
    configs = setup_configs()
    data = configs["data_loader"].load_data()
    
    
    configs = update_configs(configs,
                             data["tr"]["x"].shape,
                             data["val"]["x"].shape)
    
    
    networks, fns = configs["net"].build_network(configs, configs["net"].build_layers(configs))
    
    
    tv = TrainVal(data, configs, fns, networks)
    tv.train()

    
    
    





