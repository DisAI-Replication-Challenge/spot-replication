from train.hf_train_peft import get_config
from train.utils import get_dataset_option
import wandb
from transformers import AutoConfig, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from adapters import AutoAdapterModel
from tqdm import tqdm
# from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit, PeftConfig, PeftModel
from prompt_tuning import PromptTuningConfig, TaskType, PromptTuningInit, get_prompt_tuning_model, PromptTuningForSeq2SeqLM
from collections import namedtuple
import os
from torch.utils.data import DataLoader
import torch

from data.preprocess import preprocess_data
from train.utils import get_optimizer
from train.sampling import proportional_mixing, all_mixing

import logging

logging.basicConfig(level=logging.INFO)


def get_model_tokenizer(model_name):
    if 'prompt' in model_name:
        # remove last /...
        model_name2 = '/'.join(model_name.split("/")[:-1])
    else:
        model_name2 = model_name

    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name2)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def reset_metrics(metrics):
    training_metrics = {}
    valid_metrics = {}
    for metric in metrics:
        for key in metric.key:
            training_metrics[f"train_{key}"] = 0.0
            valid_metrics[f"valid_{key}"] = 0.0

    return training_metrics, valid_metrics


def get_possible_langs(dataloaders):
    possible_langs = []
    for dataloader in dataloaders:
        languages = dataloader.supported_languages()
        possible_langs.extend(languages)

    possible_langs = list(set(possible_langs))

    return possible_langs


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze_parameters(model, name):
    for module_name, module in model.named_children():
        if module_name == name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            unfreeze_parameters(module, name)


def get_promptinit(config):
    # if config.init_type == 'sampled':
    #     return PromptTuningInit.SAMPLED
    # el
    if config.init_type == 'text':
        return PromptTuningInit.TEXT
    elif config.init_type == 'random':
        return PromptTuningInit.RANDOM
    # elif config.init_type == 'class':
    #     return PromptTuningInit.CLASS


def get_max_target_length(dataloaders, tokenizer, max_length):
    max_lengths = []
    for dataloader in dataloaders:
        max_lengths.append(
            dataloader.get_max_target_length(tokenizer, max_length))

    return max(max_lengths)


class LanguageTaskTrainer:
    def __init__(
        self,
        config_path="../configs/config.yaml",
        wandb_project="t5-multirc-finetune",
        wandb_log_model="checkpoint",
        lanugage_adapter="adapter",  # adapter or prompt
        task_adapter="adapter",  # adapter or prompt
        training='language',
        task_name=None,
    ):
        self.config_path = config_path
        self.wandb_project = wandb_project
        self.wandb_log_model = wandb_log_model
        self.language_adapter = lanugage_adapter
        self.task_adapter = task_adapter
        self.training = training
        self.task_name = task_name

        self.config = get_config(self.config_path)
        self.config = namedtuple('config', self.config.keys())(
            *self.config.values())

    def create_model_dir(self):
        os.makedirs(self.config.output_path, exist_ok=True)

        if self.training == 'language':
            os.makedirs(
                f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}', exist_ok=True)
            return f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}'
        else:
            os.makedirs(
                f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}/{self.task_name}_{self.task_adapter}', exist_ok=True)
            return f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}/{self.task_name}_{self.task_adapter}'

    def save_model(self, model, tokenizer, prompts, total_steps, best=False):
        config = self.config

        if best:
            # path = f'{config.output_path}/{config.model_name.split("/")[-1]}/language_{self.lanugage_adapter}/task_{self.task_adapter}/{dataloader.name}-{self.config["language"]}/best_model'
            path = f'{self.create_model_dir()}/best_model'
        else:
            # path = f'{config.output_path}/{config.model_name.split("/")[-1]}/language_{self.lanugage_adapter}/task_{self.task_adapter}/{dataloader.name}-{self.config["language"]}/checkpoint_{total_steps}'
            path = f'{self.create_model_dir()}/checkpoint_{total_steps}'

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        if self.training == 'language':
            if self.language_adapter == 'adapter':
                model.save_adapter(path, f'{config.language}_adapter')
            elif self.language_adapter == 'prompt':
                torch.save(prompts[0], f'{path}/{config.language}.pt')
        else:
            if self.language_adapter == 'adapter' and self.task_adapter == 'adapter':
                model.save_adapter(path, f'{config.language}_adapter')
                model.save_adapter(path, f'{self.task_name}_adapter')
            elif self.language_adapter == 'adapter' and self.task_adapter == 'prompt':
                model.save_adapter(path, f'{config.language}_adapter')
                torch.save(prompts[0], f'{path}/{self.task_name}.pt')
            elif self.language_adapter == 'prompt' and self.task_adapter == 'adapter':
                torch.save(prompts[0], f'{path}/{config.language}.pt')
                model.save_adapter(path, f'{self.task_name}_adapter')
            elif self.language_adapter == 'prompt' and self.task_adapter == 'prompt':
                torch.save(prompts[0], f'{path}/{config.language}.pt')
                torch.save(prompts[1], f'{path}/{self.task_name}.pt')

    def train(self):
        DatasetOption = get_dataset_option(self.config.model_name)

        language = self.config.language

        if self.training == 'language':
            dataloader = DatasetOption.get('wiki')

            dataloaders = [dataloader(language=language)]

            max_token_length = self.config.max_length

        elif self.training == 'task':
            dataloaders = DatasetOption.get_language_task(
                language=language, task=self.task_name)

            dataloaders = [dataloader(language=language)
                           for dataloader in dataloaders]

            max_token_length = self.config.max_length

        wandb.init(project=self.wandb_project, name=self.wandb_log_model)
        # convert namedtuple into dict
        config = dict(self.config._asdict())
        wandb.config.update(config)

        if self.training == 'language':
            logging.info(
                f"Training language adapter for {self.config.language}")
            model, tokenizer = get_model_tokenizer(self.config.model_name)
            if self.language_adapter == 'adapter':
                model.add_adapter(f'{self.config.language}_adapter')
                model = freeze_parameters(model)
                unfreeze_parameters(model, f'{self.config.language}_adapter')
                wandb.watch(model, log="all")

            elif self.language_adapter == 'prompt':
                peft_config = PromptTuningConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    prompt_tuning_init=get_promptinit(self.config),
                    num_virtual_tokens=self.config.num_virtual_tokens,
                    tokenizer_name_or_path=self.config.model_name,
                    prompt_tuning_init_text=self.config.prompt_init_text
                )

                model = get_peft_model(
                    model, peft_config=peft_config, adapter_name=f'{self.config.language}_prompt')
                model = freeze_parameters(model)
                unfreeze_parameters(model, f'{self.config.language}_prompt')
                wandb.watch(model, log="all")

            else:
                raise ValueError(
                    "Language adapter should be either adapter or prompt")
        else:
            if self.language_adapter == 'adapter' and self.task_adapter == 'adapter':
                # load language adapter from the disk
                model, tokenizer = get_model_tokenizer(
                    f'{self.config.language_adapter}')
                model.load_adapter(f'{self.config.language_adapter}')
                model.add_adapter(f'{self.task_name}_adapter')

                model = freeze_parameters(model)
                unfreeze_parameters(model, f'{self.task_name}_adapter')
            elif self.language_adapter == 'adapter' and self.task_adapter == 'prompt':
                model, tokenizer = get_model_tokenizer(
                    f'{self.config.language_adapter}')
                model.load_adapter(f'{self.config.language_adapter}')
                peft_task_config = PromptTuningConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    prompt_tuning_init=get_promptinit(self.config),
                    num_virtual_tokens=self.config.num_virtual_tokens,
                    tokenizer_name_or_path=self.config.model_name,
                    prompt_tuning_init_text=self.config.prompt_init_text
                )

                model = get_prompt_tuning_model(
                    model, peft_config=peft_task_config, adapter_name=f'{self.task_name}_prompt')
                model = freeze_parameters(model)
                unfreeze_parameters(model, f'{self.task_name}_prompt')

            elif self.language_adapter == 'prompt' and self.task_adapter == 'adapter':
                peft_lang_config = PromptTuningConfig.from_pretrained(
                    f'{self.config.language_adapter}')

                model, tokenizer = get_model_tokenizer(
                    peft_lang_config.base_model_name_or_path)

                model = PromptTuningForSeq2SeqLM.from_pretrained(
                    model, f'{self.config.language_adapter}')

                model.add_adapter(f'{self.task_name}_adapter')
                model = freeze_parameters(model)
                unfreeze_parameters(model, f'{self.task_name}_adapter')

            elif self.language_adapter == 'prompt' and self.task_adapter == 'prompt':
                peft_lang_config = PromptTuningConfig.from_pretrained(
                    f'{self.config.language_adapter}')

                peft_task_config = PromptTuningConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    prompt_tuning_init=get_promptinit(self.config),
                    num_virtual_tokens=self.config.num_virtual_tokens,
                    tokenizer_name_or_path=self.config.model_name,
                    prompt_tuning_init_text=self.config.prompt_init_text
                )

                model, tokenizer = get_model_tokenizer(
                    peft_lang_config.base_model_name_or_path)

                model = PromptTuningForSeq2SeqLM.from_pretrained(
                    model, f'{self.config.language_adapter}', adapter_name=f'{language}_prompt')

                model = get_prompt_tuning_model(
                    model, peft_config=peft_task_config, adapter_name=f'{self.task_name}_prompt')

                model = freeze_parameters(model)
                unfreeze_parameters(model, f'{self.task_name}_prompt')

        max_token_length = get_max_target_length(
            dataloaders, tokenizer, max_token_length)

        postprocess_func = dataloaders[0].postprocess

        preprocessed_data = [
            preprocess_data(
                data, tokenizer, padding=self.config.padding, truncation=self.config.truncation, max_target_length=max_token_length, test_set=False)
            for data in dataloaders
        ]
        metrics = [
            dataloader.metrics
            for dataloader in dataloaders
        ]
        # explode the list of metrics
        metrics = [item for sublist in metrics for item in sublist]

        train_data = [data[0] for data in preprocessed_data]
        valid_data = [data[1] for data in preprocessed_data]

        print('PROPORTIONAL MIXING')
        train_data = proportional_mixing(train_data, required_size=524_288)
        print(len(train_data))
        valid_data = proportional_mixing(valid_data, round(524_288 * 0.2))
        print(len(valid_data))

        print(train_data[0])

        train_dataloader = DataLoader(
            train_data, shuffle=True, collate_fn=default_data_collator, batch_size=self.config.batch_size, pin_memory=True
        )

        eval_dataloader = DataLoader(
            valid_data, collate_fn=default_data_collator, batch_size=self.config.batch_size, pin_memory=True)

        optimizer = get_optimizer(self.config, model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.training_steps,
        )

        best_eval_loss = float("inf")

        total_steps = 0
        total_loss = 0
        training_metrics, valid_metrics = reset_metrics(metrics)

        print(torch.cuda.memory_summary())

        while True:
            model.train()

            for batch in tqdm(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss
                total_loss += loss.detach().float()

                outputs_logits = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                    max_new_tokens=max_token_length
                )

                decoded_labels, decoded_preds = postprocess_func(
                    labels=batch["labels"].detach().cpu().numpy(),
                    preds=outputs_logits.detach().cpu().numpy(),
                    tokenizer=tokenizer,
                )

                for metric in metrics:
                    value = metric.compute(decoded_labels, decoded_preds)
                    for key in metric.key:
                        training_metrics[f"train_{key}"] += value[key]

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_steps += 1

                if total_steps % self.config.eval_steps == 0 or total_steps == self.config.training_steps:
                    model.eval()
                    eval_loss = 0

                    with torch.no_grad():
                        for eval_batch in tqdm(eval_dataloader):
                            eval_batch = {k: v.to(device)
                                          for k, v in eval_batch.items()}
                            eval_outputs = model(
                                input_ids=eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], labels=eval_batch["labels"])
                            eval_loss += eval_outputs.loss.detach().float()

                            outputs_eval_logits = model.generate(
                                input_ids=eval_batch["input_ids"],
                                attention_mask=eval_batch["attention_mask"],
                                labels=eval_batch["labels"],
                                return_dict=True,
                                max_new_tokens=max_token_length
                            )

                            decoded_labels, decoded_preds = postprocess_func(
                                labels=eval_batch["labels"].detach(
                                ).cpu().numpy(),
                                preds=outputs_eval_logits.detach().cpu().numpy(),
                                tokenizer=tokenizer,
                            )

                            for metric in metrics:
                                value = metric.compute(
                                    decoded_labels, decoded_preds)
                                for key in metric.key:
                                    valid_metrics[f"valid_{key}"] += value[key]

                    print(decoded_labels, decoded_preds)
                    if total_steps == self.config.training_steps:
                        total_train_loss = total_loss / \
                            (total_steps % self.config.eval_steps)
                    else:
                        total_train_loss = total_loss / self.config.eval_steps

                    total_eval_loss = eval_loss / len(eval_dataloader)
                    total_loss = 0

                    # log into wandb
                    wandb_log = {
                        "train_loss": total_train_loss,
                        "eval_loss": total_eval_loss,
                    }

                    for metric in metrics:
                        for key in metric.key:
                            if total_steps == self.config.training_steps:
                                wandb_log[f"train_{key}"] = training_metrics[f"train_{key}"] / \
                                    (total_steps % self.config.eval_steps)
                            else:
                                wandb_log[f"train_{key}"] = training_metrics[f"train_{key}"] / \
                                    self.config.eval_steps

                            wandb_log[f"valid_{key}"] = valid_metrics[f"valid_{key}"] / len(
                                eval_dataloader)

                    training_metrics, valid_metrics = reset_metrics(metrics)

                    wandb.log(wandb_log, commit=True)
                    # print loss and metrics as one print statement
                    print(
                        f"Steps: {total_steps}, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}, {wandb_log}")

                    prompts = self.get_prompts(model)
                    # save model based on the total_steps count
                    if total_steps % self.config.save_steps == 0:
                        self.save_model(
                            model, tokenizer, prompts, total_steps)
                    # save only the best model
                    if total_eval_loss < best_eval_loss:
                        wandb.run.summary["best_eval_loss"] = total_eval_loss
                        best_eval_loss = total_eval_loss
                        self.save_model(
                            model, tokenizer, prompts, total_steps, best=True)

                    if total_steps == self.config.training_steps:
                        return

    def get_prompts(self, model):
        if self.training == 'language' and self.language_adapter == 'prompt':
            return [
                model.get_prompt_embedding_to_save(
                    adapter_name=f'{self.config.language}_prompt')
            ]
        elif self.language_adapter == 'prompt' and self.task_adapter == 'adapter':
            return [
                model.get_prompt_embedding_to_save(
                    adapter_name=f'{self.config.language}_prompt')
            ]
        elif self.language_adapter == 'prompt' and self.task_adapter == 'prompt':
            return [
                model.get_prompt_embedding_to_save(
                    adapter_name=f'{self.config.language}_prompt'),
                model.get_prompt_embedding_to_save(
                    adapter_name=f'{self.task_name}_prompt')
            ]
        elif self.language_adapter == 'adapter' and self.task_adapter == 'prompt':
            return [
                model.get_prompt_embedding_to_save(
                    adapter_name=f'{self.task_name}_prompt')
            ]
        else:
            return None

    def evaluate(self, dataset):
        DatasetOption = get_dataset_option(self.config.model_name)

        language = self.config.language

        if self.training == 'language':
            dataloader = DatasetOption.get('wiki')

            dataloaders = [dataloader(language=language)]

        elif self.training == 'task':
            dataloaders = DatasetOption.get_task(dataset)

            dataloader_list = []

            for dataloader in dataloaders:
                languages = dataloader.supported_languages()
                for language in languages:
                    dataloader_list.append(dataloader(
                        language=language))  # check this

            dataloaders = dataloader_list

        if self.language_adapter == 'adapter' and self.task_adapter == 'adapter':
            # load language adapter from the disk
            model, tokenizer = get_model_tokenizer(
                f'{self.config.language_adapter}')
            model.load_adapter(f'{self.config.language_adapter}')
            model.load_adapter(f'{self.config.task_adapter}')

        elif self.language_adapter == 'adapter' and self.task_adapter == 'prompt':
            model, tokenizer = get_model_tokenizer(
                f'{self.config.language_adapter}')
            model.load_adapter(f'{self.config.language_adapter}')
            peft_task_config = PromptTuningConfig.from_pretrained(
                f'{self.config.task_adapter}')

            model = get_prompt_tuning_model(
                model, peft_config=peft_task_config, adapter_name=f'{self.task_name}_prompt')

        elif self.language_adapter == 'prompt' and self.task_adapter == 'adapter':
            peft_lang_config = PromptTuningConfig.from_pretrained(
                f'{self.config.language_adapter}')

            model, tokenizer = get_model_tokenizer(
                peft_lang_config.base_model_name_or_path)

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, f'{self.config.language_adapter}')

            model.load_adapter(f'{self.config.task_adapter}')

        elif self.language_adapter == 'prompt' and self.task_adapter == 'prompt':
            peft_lang_config = PromptTuningConfig.from_pretrained(
                f'{self.config.language_adapter}')

            peft_task_config = PromptTuningConfig.from_pretrained(
                f'{self.config.task_adapter}')

            model, tokenizer = get_model_tokenizer(
                peft_lang_config.base_model_name_or_path)

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, f'{self.config.language_adapter}')

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, f'{self.config.task_adapter}')

        max_token_length = get_max_target_length(dataloaders, tokenizer, 128)

        postprocess_func = dataloaders[0].postprocess

        preprocessed_data = [
            preprocess_data(
                data, tokenizer, padding=self.config.padding, truncation=self.config.truncation, max_target_length=max_token_length)
            for data in dataloaders
        ]

        test_data = [data[1] for data in preprocessed_data]
        test_data = all_mixing(test_data)

        test_dataloader = DataLoader(
            test_data, collate_fn=default_data_collator, batch_size=self.config.batch_size, pin_memory=True, shuffle=True)

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        metrics = []

        test_metrics = {}
        for metric in metrics:
            for key in metric.key:
                test_metrics[f"test_{key}"] = 0.0

        model.eval()
        data_length = len(test_dataloader)
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         labels=batch["labels"], return_dict=True)

                decoded_labels, decoded_preds = postprocess_func(
                    labels=batch["labels"].detach().cpu().numpy(),
                    preds=outputs.detach().cpu().numpy(),
                    tokenizer=tokenizer,
                )

                for metric in metrics:
                    value = metric.compute(decoded_labels, decoded_preds)
                    # check if value is nan
                    if value != value:
                        data_length -= 1
                        continue
                    for key in metric.key:
                        test_metrics[f"test_{key}"] += value[key]

            print(decoded_labels[:10], decoded_preds[:10])

        for metric in metrics:
            for key in metric.key:
                test_metrics[f"test_{key}"] = test_metrics[f"test_{key}"] / \
                    data_length

        return test_metrics
