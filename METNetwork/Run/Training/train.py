"""
Main callable script to train METNet
"""

from pickle import load
from pathlib import Path

from mattstools.trainer import Trainer
from mattstools.utils import save_yaml_files, print_dict

from METNetwork.resources.utils import get_configs
from METNetwork.resources.datasets import StreamMETDataset
from METNetwork.resources.networks import METNet

import torch as T


## Manual seed for reproducibility
T.manual_seed(42)


def main():
    """Run the script"""

    ## Collect the session arguments, returning the configuration dictionaries
    data_conf, net_conf, train_conf = get_configs()

    ## Load the METNet Iterative datasets
    train_set = StreamMETDataset(**data_conf)

    ## Get the data dimensions and processing info
    do_rot = train_set.do_rot
    net_conf["base_kwargs"]["inpt_dim"] = len(train_set.inpt_list)
    net_conf["base_kwargs"]["outp_dim"] = 2

    ## Create the network
    network = METNet(do_rot, **net_conf)

    ## Save the preprocessing buffers on the network
    network.set_preproc(train_set.get_preprocess_info())

    ## Load the trainer, do this before saving incase any error arise
    trainer = Trainer(network, train_set, **train_conf)

    ## Create the save folder for the network and store the configuration dicts
    save_yaml_files(
        Path(network.full_name, "config"),
        ["data", "netw", "train"],
        [data_conf, net_conf, train_conf],
    )

    ## Run the training looop
    # trainer.explode_learning()
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
