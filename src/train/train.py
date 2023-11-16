from functools import partial
import os
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch
from torch.utils.data import DataLoader
from train.hf_train_peft import get_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml
import wandb

from data.preprocess import preprocess_data
from train.utils import get_optimizer


def get_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def train_loop(config, dataloader, metrics):

    with wandb.init(config=config):
        config = wandb.config

        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text=config.prompt_init_text,
            tokenizer_name_or_path=config.model_name,
        )

        model, tokenizer = get_model_tokenizer(config.model_name)
        model = get_peft_model(model, peft_config)
        wandb.watch(model, log="all", log_freq=100)

        train_data, valid_data, _ = preprocess_data(
            dataloader, tokenizer, test_set=False)

        train_dataloader = DataLoader(
            train_data, shuffle=True, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True
        )
        eval_dataloader = DataLoader(
            valid_data, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

        optimizer = get_optimizer(config, model)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.epochs),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        best_eval_loss = float("inf")

        for epoch in tqdm(range(config.epochs)):
            model.train()
            total_loss = 0
            eval_loss = 0

            training_metrics = {}
            valid_metrics = {}
            for metric in metrics:
                training_metrics[f"train_{metric.name}"] = 0.0
                valid_metrics[f"valid_{metric.name}"] = 0.0

            for batch in tqdm(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss
                total_loss += loss.detach().float()

                for metric in metrics:
                    metric.compute.to(device)
                    prediction = torch.argmax(outputs.logits, dim=-1)
                    value = metric.compute(batch["labels"], prediction)
                    training_metrics[f"train_{metric.name}"] += value.detach().float()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                    loss = outputs.loss
                    eval_loss += loss.detach().float()

                    decoded_preds = tokenizer.batch_decode(
                        outputs.detach().cpu().numpy(), skip_special_tokens=True)
                    decoded_preds = [text.strip() for text in decoded_preds]

                    decoded_labels = tokenizer.batch_decode(
                        batch["labels"].detach().cpu().numpy(), skip_special_tokens=True)
                    decoded_labels = [text.strip() for text in decoded_labels]

                    for metric in metrics:
                        metric.compute.to(device)
                        valid_metrics[f"valid_{metric.name}"] += metric.compute(
                            decoded_labels, decoded_preds).detach().float()

            total_train_loss = total_loss / len(train_dataloader)
            total_eval_loss = eval_loss / len(eval_dataloader)

            # log into wandb
            wandb_log = {
                "epoch": epoch,
                "train_loss": total_train_loss,
                "eval_loss": total_eval_loss,
            }

            for metric in metrics:
                wandb_log[f"train_{metric.name}"] = training_metrics[f"train_{metric.name}"] / len(
                    train_dataloader)
                wandb_log[f"valid_{metric.name}"] = valid_metrics[f"valid_{metric.name}"] / len(
                    eval_dataloader)

            wandb.log(wandb_log, commit=True)
            # print loss and metrics as one print statement
            print(
                f"Epoch: {epoch}, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}, {wandb_log}")
            # save only the best model
            if total_eval_loss < best_eval_loss:
                wandb.run.summary["best_eval_loss"] = total_eval_loss
                best_eval_loss = total_eval_loss
                model.save_pretrained(config.output_path)
                tokenizer.save_pretrained(config.output_path)


def train_with_sweeps(dataloader, metrics, config_path, wandb_project="t5-multirc-finetune", wandb_log_model="checkpoint"):
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # use wandb sweeps
    sweep_config = get_config(config_path)
    sweep_config['parameters']['wandb_project'] = wandb_project

    sweep_id = wandb.sweep(sweep_config, project=wandb_project)
    wandb.agent(sweep_id, lambda: partial(
        train_loop, dataloader=dataloader, metrics=metrics))


def train(dataloader, metrics, config_path, wandb_project="t5-multirc-finetune", wandb_log_model="checkpoint"):
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # use wandb sweeps
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_loop(config, dataloader, metrics)
