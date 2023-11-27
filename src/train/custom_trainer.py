import os
import wandb

from train.hf_train_peft import get_config, train as hf_train_peft
from train.train import train_with_sweeps
from train.train import train as custom_train
from data.t5.task import DatasetOption
from eval.hf_inference_peft import inference
from collections import namedtuple


class CustomTrainer:
    def __init__(self, use_sweep=False, use_hf=False, config_path="../configs/config.yaml", wandb_project="t5-multirc-finetune", wandb_log_model="checkpoint"):
        self.use_sweep = use_sweep
        self.use_hf = use_hf
        self.wandb_project = wandb_project
        self.wandb_log_model = wandb_log_model
        self.config_path = config_path

    def train(self, dataset):

        dataloader = DatasetOption.get(dataset)()
        metrics = dataloader.metrics

        if self.use_hf:
            hf_train_peft(dataloader, metrics, self.config_path,
                          self.wandb_project, self.wandb_log_model)
        else:
            if self.use_sweep:
                train_with_sweeps(dataloader, metrics, self.config_path,
                                  self.wandb_project, self.wandb_log_model)
            else:
                custom_train(dataloader, metrics, self.config_path,
                             self.wandb_project, self.wandb_log_model)

    def evaluate(self, dataset):
        os.environ["WANDB_PROJECT"] = self.wandb_project
        os.environ["WANDB_LOG_MODEL"] = self.wandb_log_model

        dataloader = DatasetOption.get(dataset)()
        metrics = dataloader.metrics

        config = get_config(self.config_path)
        config = namedtuple('config', config.keys())(*config.values())
        results = inference(config, dataloader, metrics)

        print(results)
