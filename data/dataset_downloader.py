from datasets import load_dataset
# import gdown
# import os
# from zipfile import ZipFile

cache_path = "./.cache"

# Missing datasets: CR, CxC, Wikilingua (gem)

# For MNLI, they used validation_matched and validation_mismatched

extended_datasets = {
    'glue': ["ax", "cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp",  "rte", "sst2", "stsb", "wnli"],
    'super_glue': ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed', 'axb', 'axg'],
    'anli': [],
    'saattrupdan/doc-nli': [],  # need to check
    'snli': [],
    'go_emotions': ['raw', 'simplified'],
    'sentiment140': [],
    'c4': ['en'],
    'squad': [],
    'newsqa': ['combined-csv'],  # need to check,
    'trivia_qa': ['rc'],  # need to check,
    'search_qa': ['train_test_val'],  # need to check,
    'hotpot_qa': ['fullwiki'],  # need to check,
    'nq_open': [],  # need to check,
    'cosmos_qa': [],
    'Rowan/hellaswag': [],
    'piqa': [],
    'social_i_qa': [],
    'winogrande': ['winogrande_xl'],
    'gem': ['common_gen', 'dart', 'e2e_nlg', 'schema_guided_dialog', 'web_nlg_en', 'wiki_auto_asset', 'xsum'],
    'drop': [],
    'wmt14': ['de-en'],
    'wmt15': ['fr-en'],
    'wmt16': ['ro-en'],
    'aeslc': [],  # need to check
    'billsum': [],
    'gigaword': [],
    'multi_news': [],  # need to check
    'newsroom': [],  # need to check
    'samsum': [],  # need to check
    'cnn_dailymail': ['3.0.0'],
    'race': ['middle'],
    'yelp_polarity': [],
}

original_datasets = {
    'glue': ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte'],  # done
    # done
    'super_glue': ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed'],
    'anli':  [],  # splits: train_r1, train_r2, train_r3 # done
    'saattrupdan/doc-nli': [],  # done
    'snli': [],  # done
    'go_emotions': ['simplified'],
    'sentiment140': [],
    # 'c4': ['en'],  # done
    'squad': [],  # done
    # 'newsqa': ['combined-json'],  # done need download manually
    'trivia_qa': ['rc'],  # done
    'search_qa': ['train_test_val'],  # done
    'hotpot_qa': ['fullwiki'],  # done
    'nq_open': [],  # done
    'cosmos_qa': [],
    'Rowan/hellaswag': [],
    'piqa': [],
    'social_i_qa': [],
    'winogrande': ['winogrande_xl'],
    'gem': ['common_gen', 'dart', 'e2e_nlg', 'schema_guided_dialog', 'web_nlg_en', 'wiki_auto_asset', 'xsum'],
    'drop': [],  # done
    'wmt14': ['de-en'],
    'wmt15': ['fr-en'],
    'wmt16': ['ro-en'],
    'aeslc': [],  # done
    'billsum': [],  # done
    'gigaword': [],  # done
    'multi_news': [],  # done
    'newsroom': [],  # done
    'samsum': [],  # done
    'cnn_dailymail': ['3.0.0'],  # done
    'race': ['middle'],  # done
    'yelp_polarity': [],
}


# def download_dataset_from_gd(url, dataset_name):
#     output_name = f'{cache_path}/{dataset_name}.zip'
#     gdown.download(url, output_name, quiet=False)
#     # unzip file using python
#     with ZipFile(output_name, 'r') as zipObj:
#         zipObj.extractall(cache_path)
#     print(f"Downloaded {dataset_name} to {output_name}")
#     # remove zip file
#     os.remove(output_name)


# gd_links = {
#     'docnli': 'https://drive.google.com/file/d/16TZBTZcb9laNKxIvgbs5nOBgq3MhND5s/view?usp=sharing',
# }


def download_datasets():
    for key, value in original_datasets.items():
        if len(value) == 0:
            dataset = load_dataset(key, cache_dir=cache_path)
        else:
            for subset in value:
                dataset = load_dataset(key, subset, cache_dir=cache_path)


if __name__ == '__main__':
    download_datasets()
