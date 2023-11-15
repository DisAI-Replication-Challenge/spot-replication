from data.preprocess import preprocess_data
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm


def load_model_tokenizer(model_name):
    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def eval_data(model, tokenizer, data, metrics):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    results = list()

    with torch.no_grad():
        for batch in tqdm(data):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["targets"], return_dict=True)

            decoded_preds = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True)
            decoded_preds = [text.strip() for text in decoded_preds]

            decoded_labels = tokenizer.batch_decode(
                batch["targets"].detach().cpu().numpy(), skip_special_tokens=True)
            decoded_labels = [text.strip() for text in decoded_labels]

            scores = dict()

            for metric in metrics:
                metric.compute.to(device)
                scores[metric.name] = metric.compute(
                    decoded_labels, decoded_preds).detach().float()

            results.append(scores)

    return results


def inference(model_name, dataloader, metrics):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, tokenizer = load_model_tokenizer(model_name)

    _, _, test_data = preprocess_data(dataloader, tokenizer, test_set=True)

    model.to(device)

    return eval_data(model, tokenizer, test_data, metrics)
