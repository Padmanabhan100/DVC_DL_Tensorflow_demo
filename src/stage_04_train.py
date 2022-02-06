from subprocess import call
from src.utils.all_utils import read_yaml,create_directory
from src.utils.callbacks import create_and_save_tensorboard_callbacks, create_and_save_checkpoint_callbacks
import argparse
import pandas as pd
import os
from tqdm import tqdm
import shutil
import logging


def train_model(config_path, params_path):
    # reading the config and params yaml file
    config = read_yaml(config_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    
if __name__ == "__main__":
    # create a ArgumentParser object
    args = argparse.ArgumentParser()

    # add arguments --config and --params on cli
    args.add_argument("--config",'-c',default='config/config.yaml')
    args.add_argument("--params",'-p',default='params.yaml')

    # call the method
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>stage-4 started")
        train_model(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info("stage-3 completed! Training is complete >>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e