from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from prompt_tuning import PromptTuningConfig, PromptTuningForSeq2SeqLM

from data.preprocess import preprocess_data
from train.sampling import all_mixing


def load_model_tokenizer(model_name):
    config = PromptTuningConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.tokenizer_name_or_path)
    model = PromptTuningForSeq2SeqLM.from_pretrained(model, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def eval_data(model, tokenizer, data, metrics, dataloader):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_metrics = {}
    for metric in metrics:
        for key in metric.key:
            test_metrics[f"test_{key}"] = 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                     labels=batch["labels"], return_dict=True)

            decoded_labels, decoded_preds = dataloader.postprocess(
                labels=batch["labels"].detach().cpu().numpy(),
                preds=outputs.detach().cpu().numpy(),
                tokenizer=tokenizer,
            )

            for metric in metrics:
                if metric.name in ["F1 over all answers", "F1 with invalid"]:
                    decoded_labels, decoded_preds = dataloader.postprocess_for_metrics(
                        decoded_labels, decoded_preds)
                value = metric.compute(decoded_labels, decoded_preds)
                for key in metric.key:
                    test_metrics[f"test_{key}"] += value[key]

    for metric in metrics:
        for key in metric.key:
            test_metrics[f"test_{key}"] = test_metrics[f"test_{key}"] / \
                len(data)

    return test_metrics


def inference(config, dataloader, metrics):
    model_name = f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataloader.name}/best_model'
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, tokenizer = load_model_tokenizer(model_name)

    if 'mixture' in dataloader.name:
        preprocessed_data = [
            preprocess_data(
                dataset, tokenizer, padding=config.padding, truncation=config.truncation)
            for dataset in dataloader.datasets
        ]
        test_data = [data[1] for data in preprocessed_data]
    else:
        _, test_data, _ = preprocess_data(dataloader, tokenizer)
        test_data = [test_data]

    test_data = all_mixing(test_data)

    test_dataloader = DataLoader(
        test_data, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

    model.to(device)

    return eval_data(model, tokenizer, test_dataloader, metrics, dataloader)
