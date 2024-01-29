from eval.cpeft_inference import load_model_tokenizer
from transformers import default_data_collator
import torch
from torch.utils.data import DataLoader
from data.preprocess import preprocess_data
import os
import pandas as pd


def inference(model_name, config, dataloader, eval_name='super_glue'):
    m_name = ''.join(model_name.split(
        '/')[2:3]).replace('/', '_').replace('best_model-', '')
    final_path = f'{config.output_path}/eval/{m_name}-{eval_name}'
    os.makedirs(f'{config.output_path}', exist_ok=True)
    os.makedirs(f'{config.output_path}/eval', exist_ok=True)
    os.makedirs(f'{final_path}', exist_ok=True)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, tokenizer = load_model_tokenizer(model_name)
    preprocessed_data = [
        preprocess_data(
            dataset, tokenizer, padding=config.padding, truncation=config.truncation)
        for dataset in dataloader.datasets
    ]
    test_data = [data[1] for data in preprocessed_data]

    for idx, data in enumerate(test_data):
        current_dataloader = dataloader.datasets[idx]
        metrics = current_dataloader.metrics

        test_metrics = {}
        for metric in metrics:
            for key in metric.key:
                test_metrics[f"test_{key}"] = 0.0

        test_dataloader = DataLoader(
            data, batch_size=config.batch_size, shuffle=False, collate_fn=default_data_collator)
        model.to(device)
        model.eval()

        data_lengths = {f'{metric.name}': len(test_dataloader)
                        for metric in metrics}
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         labels=batch["labels"], return_dict=True, max_length=128)

                decoded_labels, decoded_preds = current_dataloader.postprocess(
                    labels=batch["labels"].detach().cpu().numpy(),
                    preds=outputs.detach().cpu().numpy(),
                    tokenizer=tokenizer,
                )

                for metric in metrics:
                    if metric.name in ["F1 with invalid"] or current_dataloader.name == 'stsb':
                        decoded_labels, decoded_preds = current_dataloader.postprocess_for_metrics(
                            decoded_labels, decoded_preds)
                    elif metric.name in ['Match all', 'Deduplicate metric', "F1 over all answers",]:
                        decoded_labels, decoded_preds = current_dataloader.postprocess_for_metrics(
                            decoded_labels, decoded_preds, batch)
                    value = metric.compute(decoded_labels, decoded_preds)
                    # check if value is nan
                    if value != value:
                        print('nan')
                        data_lengths[f'{metric.name}'] -= 1
                        continue
                    for key in metric.key:
                        test_metrics[f"test_{key}"] += value[key]

        print(decoded_labels[:10], decoded_preds[:10])
        for metric in metrics:
            for key in metric.key:
                test_metrics[f"test_{key}"] = test_metrics[f"test_{key}"] / \
                    data_lengths[f'{metric.name}']

        print(test_metrics)
        df = pd.DataFrame(list(test_metrics.items()),
                          columns=['Metric', 'Value'])
        df.to_csv(f'{final_path}/{current_dataloader.name}.csv', index=False)
