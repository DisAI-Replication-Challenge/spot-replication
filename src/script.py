import argparse
import os

from train.custom_trainer import CustomTrainer
from utils import get_wandb_config, get_huggingface_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser('SPoT Replication')

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
    parser.add_argument('--mixture', action='store_true',
                        help='Use mixture of datasets')
    args = parser.parse_args()

    wandb_config = get_wandb_config("../config/wandb.conf")
    huggingface_config = get_huggingface_config("../config/huggingface.conf")

    os.environ["WANDB_API_KEY"] = wandb_config.WANDB_API_KEY
    os.environ["WANDB_USERNAME"] = wandb_config.WANDB_USERNAME
    os.environ["WANDB_DIR"] = wandb_config.WANDB_DIR
    os.environ['HF_API_KEY'] = huggingface_config.HF_API_KEY

    trainer = CustomTrainer(
        args.use_sweep,
        args.use_hf,
        args.use_cpeft,
        args.config_path,
        args.wandb_project,
        args.wandb_log_model
    )

    trainer.train(args.dataset, args.mixture)

    trainer.evaluate(args.dataset, args.mixture)
