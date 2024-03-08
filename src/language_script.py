import argparse
import os

# from train.custom_trainer import CustomTrainer
from utils import get_wandb_config, get_huggingface_config
from train.language_trainer import LanguageTaskTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'SPoT Replication - Language & Task transferability')

    parser.add_argument('--config_path', type=str, default='./configs/language_configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--wandb_project', type=str, default='t5-multirc-finetune',
                        help='Wandb project name')
    parser.add_argument('--wandb_log_model', type=str, default='checkpoint',
                        help='Wandb log model')
    parser.add_argument('--language_adapter',
                        choices=['adapter', 'prompt'], default='adapter')
    parser.add_argument(
        '--task_adapter', choices=['adapter', 'prompt'], default='adapter')
    parser.add_argument(
        '--training', choices=['language', 'task'], default='hf')
    parser.add_argument('--task_name', type=str,
                        default=None)

    args = parser.parse_args()

    wandb_config = get_wandb_config("../config/wandb.conf")
    huggingface_config = get_huggingface_config("../config/huggingface.conf")

    os.environ["WANDB_API_KEY"] = wandb_config.WANDB_API_KEY
    os.environ["WANDB_USERNAME"] = wandb_config.WANDB_USERNAME
    os.environ["WANDB_DIR"] = wandb_config.WANDB_DIR
    os.environ['HF_API_KEY'] = huggingface_config.HF_API_KEY

    trainer = LanguageTaskTrainer(
        config_path=args.config_path,
        wandb_project=args.wandb_project,
        wandb_log_model=args.wandb_log_model,
        lanugage_adapter=args.language_adapter,
        task_adapter=args.task_adapter,
        training=args.training,
        task_name=args.task_name
    )

    trainer.train()

    # print(trainer.evaluate(args.dataset))
