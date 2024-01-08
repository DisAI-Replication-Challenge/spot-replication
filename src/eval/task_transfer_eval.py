from eval.cpeft_inference import load_model_tokenizer
from transformers import default_data_collator
import torch
from torch.utils.data import DataLoader
from data.preprocess import preprocess_data
import os
import pandas as pd


def inference(model_name, config, dataloader, eval_name):
    m_name = ''.join(model_name.split(
        '/')[2:3]).replace('/', '_').replace('best_model-', '')
    final_path = f'{config.output_path}/eval/{m_name}-{eval_name}'
    os.makedirs(f'{config.output_path}', exist_ok=True)
    os.makedirs(f'{config.output_path}/eval', exist_ok=True)
    os.makedirs(f'{final_path}', exist_ok=True)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, tokenizer = load_model_tokenizer(model_name)
    _, test_data, _ = preprocess_data(
        dataloader, tokenizer, padding=config.padding, truncation=config.truncation)

    metrics = dataloader.metrics

    test_metrics = {}
    for metric in metrics:
        for key in metric.key:
            test_metrics[f"test_{key}"] = 0.0

    test_dataloader = DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False, collate_fn=default_data_collator)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                     labels=batch["labels"], return_dict=True, max_length=128)

            decoded_labels, decoded_preds = dataloader.postprocess(
                labels=batch["labels"].detach().cpu().numpy(),
                preds=outputs.detach().cpu().numpy(),
                tokenizer=tokenizer,
            )

            for metric in metrics:
                if metric.name in ["F1 with invalid"] or dataloader.name == 'stsb':
                    decoded_labels, decoded_preds = dataloader.postprocess_for_metrics(
                        decoded_labels, decoded_preds)
                elif metric.name in ['Match all', 'Deduplicate metric', "F1 over all answers",]:
                    decoded_labels, decoded_preds = dataloader.postprocess_for_metrics(
                        decoded_labels, decoded_preds, batch)

                value = metric.compute(decoded_labels, decoded_preds)
                for key in metric.key:
                    test_metrics[f"test_{key}"] += value[key]

        for metric in metrics:
            for key in metric.key:
                test_metrics[f"test_{key}"] = test_metrics[f"test_{key}"] / \
                    len(test_dataloader)

        print(test_metrics)
        df = pd.DataFrame(list(test_metrics.items()),
                          columns=['Metric', 'Value'])
        df.to_csv(f'{final_path}/{dataloader.name}.csv', index=False)
