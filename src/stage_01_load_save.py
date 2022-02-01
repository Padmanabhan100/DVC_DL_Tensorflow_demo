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
create_directory([log_dir])
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), 
                    level=logging.INFO, format=logging_str, filemode='a')

# create a function to copy file from one source path to another destination path
def copy_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    for file in tqdm(list_of_files, total=N, desc=f'copying file from {source_download_dir} to {local_data_dir}',colour='green'):
        src = os.path.join(source_download_dir,file)
        dest = os.path.join(local_data_dir,file)
        shutil.copy(src,dest)

def get_data(config_path):
    config = read_yaml(config_path)

    source_download_dirs = config['source_download_dirs']
    local_data_dirs = config['local_data_dirs']

    for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs, local_data_dirs), total=2, desc="List of folders", colour='red'):
        create_directory([local_data_dir])
        copy_file(source_download_dir, local_data_dir)


if __name__ == "__main__":
    # create a ArgumentParser object
    args = argparse.ArgumentParser()

    # add a argument --config on cli
    args.add_argument("--config",'-c',default='config/config.yaml')

    # call the method
    parsed_args = args.parse_args()

    try:
        logging.info("stage-1 started")
        get_data(config_path = parsed_args.config)
        logging.info("stage-1 completed! all the data are saved in local")
    except Exception as e:
        logging.exception(e)
        raise e