"""
Miscellaneous utility functions for the METNetwork package
"""

from pathlib import Path
from typing import Tuple
import argparse

from mattstools.utils import str2bool, load_yaml_files, args_into_conf


def get_configs() -> Tuple[dict, dict, dict]:
    """Loads, modifies, and returns three configuration dictionaries using command
    line arguments
    - For configuring the dataset, network and training scheme
    - One can set the names of the config files to load
    - One can then modify the returned dictionaries using additional arguments
    """

    parser = argparse.ArgumentParser()

    ## Config files
    parser.add_argument(
        "--data_conf",
        type=str,
        default="./config/data.yaml",
        help="The config file to use for data setup",
    )
    parser.add_argument(
        "--net_conf",
        type=str,
        default="./config/netw.yaml",
        help="The config file to use for network setup",
    )
    parser.add_argument(
        "--train_conf",
        type=str,
        default="./config/train.yaml",
        help="The config file to use for training scheme",
    )

    ## Data preparation
    parser.add_argument(
        "--scaler_nm",
        type=str,
        help="Name of preprocessing scaler",
    )
    parser.add_argument(
        "--lep_vars",
        type=str,
        help="Name of the lepton and MET input coordinates seperated by commas",
    )
    parser.add_argument(
        "--jet_vars",
        type=str,
        help="Name of the jets input coordinates seperated by commas",
    )
    parser.add_argument(
        "--out_vars",
        type=str,
        help="Name of the output/target coordinates seperated by commas",
    )

    ## Network base kwargs
    parser.add_argument(
        "--name", type=str, help="The name to use for saving the network"
    )
    parser.add_argument(
        "--save_dir", type=str, help="The directory to use for saving the network"
    )

    ## Learning scheme
    parser.add_argument(
        "--resume",
        type=str2bool,
        help="Resume the latest training checkpoint",
        nargs="?",
        const=True,
        default=False,
    )

    ## Load the arguments
    args = parser.parse_args()

    ## Load previous configs if resuming, otherwise keep defaults
    if args.resume:
        args.data_conf = Path(args.save_dir, args.name, "config/data.yaml")
        args.net_conf = Path(args.save_dir, args.name, "config/netw.yaml")
        args.train_conf = Path(args.save_dir, args.name, "config/train.yaml")

    ## Load the config dictionaries
    data_conf, net_conf, train_conf = load_yaml_files(
        [args.data_conf, args.net_conf, args.train_conf]
    )

    ## Some arguments are identified by the exact keys in each dict

    ## Other arguments need more manual placement in the configuration dicts
    args_into_conf(args, net_conf, "name", "base_kwargs/name")
    args_into_conf(args, net_conf, "save_dir", "base_kwargs/save_dir")

    return data_conf, net_conf, train_conf
