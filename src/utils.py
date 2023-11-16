import configparser as ConfigParser
from collections import namedtuple


WandbConfig = namedtuple(
    'WandbConfig', ['WANDB_API_KEY', 'WANDB_USERNAME', 'WANDB_DIR'])


def get_wandb_config(path):
    config = ConfigParser.ConfigParser()
    config.read(path)

    return WandbConfig(
        WANDB_API_KEY=config.get('wandb', 'WANDB_API_KEY'),
        WANDB_USERNAME=config.get('wandb', 'WANDB_USERNAME'),
        WANDB_DIR=config.get('wandb', 'WANDB_DIR')
    )
