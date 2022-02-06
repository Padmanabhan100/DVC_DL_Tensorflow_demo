from src.utils.all_utils import read_yaml,create_directory
import argparse
import pandas as pd
import os
from tqdm import tqdm
import shutil
import logging

# create a log string and create a log directory
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), 
                    level=logging.INFO, format=logging_str, filemode='a')

def prepare_callbacks(config_path, params_path):
    pass

if __name__ == "__main__":
    # create a ArgumentParser object
    args = argparse.ArgumentParser()

    # add arguments --config and --params on cli
    args.add_argument("--config",'-c',default='config/config.yaml')
    args.add_argument("--params",'-p',default='params.yaml')

    # call the method
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>stage-3 started")
        prepare_callbacks(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info("stage-3 completed! callbacks are prepared and saved as binary >>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e