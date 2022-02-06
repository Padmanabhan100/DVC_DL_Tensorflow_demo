from src.utils.all_utils import read_yaml,create_directory
from src.utils.models import get_VGG_16_model, prepare_model
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


def prepare_base_model(config_path,params_path):
    # read the configuration yaml file
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = config['artifacts']["ARTIFACTS_DIR"]

    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_name = artifacts['BASE_MODEL_NAME']
    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)
    create_directory([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path,base_model_name)
   
    model = get_VGG_16_model(input_shape=params['IMAGE_SIZE'], model_path=base_model_path)

    full_model = prepare_model(
        model,
        CLASSES=params['CLASSES'],
        freeze_all=True,
        freeze_till=None,
        learning_rate=params['LEARNING_RATE']
    )

    updated_base_model_path = os.path.join(base_model_dir_path,artifacts['UPDATED_BASE_MODEL_NAME'])
    logging.info(f"{model.summary()}")

    full_model.save(updated_base_model_path)

if __name__ == "__main__":
    # create a ArgumentParser object
    args = argparse.ArgumentParser()

    # add arguments --config and --params on cli
    args.add_argument("--config",'-c',default='config/config.yaml')
    args.add_argument("--params",'-p',default='params.yaml')

    # call the method
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>stage-2 started")
        prepare_base_model(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info("stage-2 completed! base model is created >>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e