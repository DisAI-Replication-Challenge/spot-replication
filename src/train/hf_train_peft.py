import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup
import os
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from collections import namedtuple
import torch

from data.preprocess import preprocess_data
from train.utils import create_arguments, get_config, get_optimizer


def get_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def train_loop(dataloader, metrics, config):

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text=config.prompt_init_text,
        tokenizer_name_or_path=config.model_name,
    )

    model, tokenizer = get_model_tokenizer(config.model_name)
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    train_data, valid_data, _ = preprocess_data(
        dataloader, tokenizer, test_set=False)

    training_args = create_arguments(len(train_data), config, metrics)

    def compute_metrics(eval_pred):
        scores = dict()
        labels_ids = eval_pred.label_ids
        pred_ids = eval_pred.predictions[0]

        decoded_preds = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        decoded_preds = [text.strip() for text in decoded_preds]

        labels_ids[labels_ids == -100] = tokenizer.pad_token_id

        decoded_labels = tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True)
        decoded_labels = [text.strip() for text in decoded_labels]

        for metric in metrics:
            scores[metric.name] = metric.compute(decoded_labels, decoded_preds)
        return scores

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    optimizer = get_optimizer(config, model)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=(len(train_data) * config.epochs),
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model()

    dataset_name = dataloader.subset if dataloader.subset is not None else dataloader.benchmark_name

    output_dir = f"outputs/{dataset_name}-t5-base-peft"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(dataloader, metrics, config_path, wandb_project="t5-multirc-finetune", wandb_log_model="checkpoint"):
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    config = get_config(config_path)
    config = namedtuple('config', config.keys())(*config.values())

    train_loop(dataloader, metrics, config)
