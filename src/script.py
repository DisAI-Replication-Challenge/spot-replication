import argparse
import os

from train.custom_trainer import CustomTrainer
from utils import get_wandb_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser('SPoT Repliation')

    parser.add_argument('--dataset', type=str, default='multirc',
                        help='Dataset to use for training')
    parser.add_argument('--use_sweep', action='store_true',
                        help='Use wandb sweeps')
    parser.add_argument('--use_hf', action='store_true',
                        help='Use huggingface trainer')
    parser.add_argument('--use_cpeft', action='store_true',
                        help='Use custom prompt tuning')
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--wandb_project', type=str, default='t5-multirc-finetune',
                        help='Wandb project name')
    parser.add_argument('--wandb_log_model', type=str, default='checkpoint',
                        help='Wandb log model')
    args = parser.parse_args()

    wandb_config = get_wandb_config("../config/wandb.conf")

    os.environ["WANDB_API_KEY"] = wandb_config.WANDB_API_KEY
    os.environ["WANDB_USERNAME"] = wandb_config.WANDB_USERNAME
    os.environ["WANDB_DIR"] = wandb_config.WANDB_DIR

    trainer = CustomTrainer(
        args.use_sweep,
        args.use_hf,
        args.use_cpeft,
        args.config_path,
        args.wandb_project,
        args.wandb_log_model
    )

    trainer.train(args.dataset)

    trainer.evaluate(args.dataset)
