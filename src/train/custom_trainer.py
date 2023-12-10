import pandas as pd

from train.hf_train_peft import get_config, train as hf_train_peft
from train.train import train_with_sweeps
from train.train import train as custom_train
from train.cpeft_train import train as cpeft_train
from train.cpeft_train import train_with_sweeps as cpeft_train_with_sweeps
from data.t5.task import DatasetOption
from eval.hf_inference_peft import inference
from eval.cpeft_inference import inference as cpeft_inference
from collections import namedtuple


class CustomTrainer:
    def __init__(
        self,
        use_sweep=False,
        use_hf=False,
        use_cpeft=False,
        config_path="../configs/config.yaml",
        wandb_project="t5-multirc-finetune",
        wandb_log_model="checkpoint"
    ):
        self.use_sweep = use_sweep
        self.use_hf = use_hf
        self.use_cpeft = use_cpeft
        self.wandb_project = wandb_project
        self.wandb_log_model = wandb_log_model
        self.config_path = config_path

    def train(self, dataset, mixture=False):

        if mixture:
            dataloader = DatasetOption.get('mixture')(dataset)
        else:
            dataloader = DatasetOption.get(dataset)()
        metrics = dataloader.metrics

        if self.use_hf:
            hf_train_peft(
                dataloader,
                metrics,
                self.config_path,
                self.wandb_project,
                self.wandb_log_model
            )
        elif self.use_cpeft:
            if self.use_sweep:
                cpeft_train_with_sweeps(
                    dataloader,
                    metrics,
                    self.config_path,
                    self.wandb_project,
                    self.wandb_log_model
                )
            else:
                cpeft_train(dataloader, metrics, self.config_path,
                            self.wandb_project, self.wandb_log_model)
        else:
            if self.use_sweep:
                train_with_sweeps(
                    dataloader,
                    metrics,
                    self.config_path,
                    self.wandb_project,
                    self.wandb_log_model
                )
            else:
                custom_train(
                    dataloader,
                    metrics,
                    self.config_path,
                    self.wandb_project,
                    self.wandb_log_model
                )

    def evaluate(self, dataset, mixture=False):
        if mixture:
            dataloader = DatasetOption.get('mixture')(dataset)
        else:
            dataloader = DatasetOption.get(dataset)()
        metrics = dataloader.metrics

        config = get_config(self.config_path)
        config = namedtuple('config', config.keys())(*config.values())
        if self.use_cpeft:
            results = cpeft_inference(config, dataloader, metrics)
        else:
            results = inference(config, dataloader, metrics)

        print(results)
        model_name = f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}'

        df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        df.to_csv(f'{model_name}/valid.csv', index=False)

        print(results)
