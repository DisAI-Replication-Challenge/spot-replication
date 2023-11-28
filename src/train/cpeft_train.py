from functools import partial
import os
import torch
from torch.utils.data import DataLoader
from train.hf_train_peft import get_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml
import wandb
from prompt_tuning import PromptTuningConfig, PromptTuningInit, TaskType, get_prompt_tuning_model


from data.preprocess import preprocess_data
from train.utils import get_optimizer


def get_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def train_loop(config, dataloader, metrics):
    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(
        f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}', exist_ok=True)

    with wandb.init(config=config):
        config = wandb.config

        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            # prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=100,
            # prompt_tuning_init_text=config.prompt_init_text,
            tokenizer_name_or_path=config.model_name,
        )

        model, tokenizer = get_model_tokenizer(config.model_name)
        model = get_prompt_tuning_model(model, peft_config)
        wandb.watch(model, log="all", log_freq=100)

        train_data, _, _ = preprocess_data(
            dataloader, tokenizer, test_set=False)
        # train test split
        loader = train_data.train_test_split(
            test_size=0.2, shuffle=True)
        train_data, valid_data = loader['train'], loader['test']

        train_dataloader = DataLoader(
            train_data, shuffle=True, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True
        )

        eval_dataloader = DataLoader(
            valid_data, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

        optimizer = get_optimizer(config, model)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.training_steps,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        best_eval_loss = float("inf")

        total_steps = 0
        while True:
            model.train()
            total_loss = 0

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

                decoded_labels, decoded_preds = dataloader.postprocess(
                    labels=batch["labels"].detach().cpu().numpy(),
                    preds=torch.argmax(
                        outputs.logits, -1).detach().cpu().numpy(),
                    tokenizer=tokenizer,
                )

                for metric in metrics:
                    if metric.name in ["F1 over all answers"]:
                        decoded_labels, decoded_preds = dataloader.postprocess_for_metrics(
                            decoded_labels, decoded_preds)
                    value = metric.compute(decoded_labels, decoded_preds)
                    training_metrics[f"train_{metric.name}"] += value[metric.key]

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_steps += 1

                if total_steps % config.eval_steps == 0 or total_steps == config.training_steps:
                    model.eval()
                    eval_loss = 0

                    with torch.no_grad():
                        for eval_batch in tqdm(eval_dataloader):
                            eval_batch = {k: v.to(device)
                                          for k, v in eval_batch.items()}
                            eval_outputs = model(
                                input_ids=eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], labels=eval_batch["labels"])
                            eval_loss += eval_outputs.loss.detach().float()

                            decoded_labels, decoded_preds = dataloader.postprocess(
                                labels=eval_batch["labels"].detach(
                                ).cpu().numpy(),
                                preds=torch.argmax(
                                    eval_outputs.logits, -1).detach().cpu().numpy(),
                                tokenizer=tokenizer,
                            )

                            for metric in metrics:
                                if metric.name in ["F1 over all answers"]:
                                    decoded_labels, decoded_preds = dataloader.postprocess_for_metrics(
                                        decoded_labels, decoded_preds)
                                valid_metrics[f"valid_{metric.name}"] += metric.compute(
                                    decoded_labels, decoded_preds)[metric.key]

                    total_train_loss = total_loss / total_steps
                    total_eval_loss = eval_loss / len(eval_dataloader)

                    # log into wandb
                    wandb_log = {
                        "train_loss": total_train_loss,
                        "eval_loss": total_eval_loss,
                    }

                    for metric in metrics:
                        wandb_log[f"train_{metric.name}"] = training_metrics[f"train_{metric.name}"] / total_steps
                        wandb_log[f"valid_{metric.name}"] = valid_metrics[f"valid_{metric.name}"] / len(
                            eval_dataloader)

                    wandb.log(wandb_log, commit=True)
                    # print loss and metrics as one print statement
                    print(
                        f"Steps: {total_steps}, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}, {wandb_log}")
                    # save model based on the total_steps count
                    model.save_pretrained(
                        f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/checkpoint_{total_steps}')
                    tokenizer.save_pretrained(
                        f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/checkpoint_{total_steps}')
                    # save prompt
                    prompt = model.get_prompt_embedding_to_save()
                    torch.save(
                        prompt, f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/checkpoint_{total_steps}/prompt.pt')
                    # save only the best model
                    if total_eval_loss < best_eval_loss:
                        wandb.run.summary["best_eval_loss"] = total_eval_loss
                        best_eval_loss = total_eval_loss
                        model.save_pretrained(
                            f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/best_model')
                        tokenizer.save_pretrained(
                            f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/best_model')
                        torch.save(
                            prompt, f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/best_model/prompt.pt')

                    if total_steps == config.training_steps:
                        return


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
