import babel
from datasets import load_dataset as hfload_dataset
import numpy as np
import pandas as pd
import regex
import re
from datasets import DatasetDict
from datasets import Dataset as HFDataset
from collections import OrderedDict, namedtuple
import collections

import data.metrics as metrics
from data.dataset import Dataset

Metric = namedtuple('Metric', ['name', 'compute', 'key'])


class CLEF2022(Dataset):
    def __init__(self, split='train', language='english'):
        self.languages = [language]
        super().__init__(None, None, split)
        self.name = 'CLEF2022'
        # self.languages = ["arabic", "bulgarian",
        #                   "dutch", "english", "spanish", "turkish"]
        self.label_names = ['unworthy', 'checkworthy']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def load_dataset(self):
        train_data = pd.DataFrame()
        valid_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for language in self.languages:
            train_df = pd.read_csv(
                f'../data/clef2022/CT22_{language}_1A_checkworthy_train.tsv', sep='\t')
            train_df['language'] = language
            train_data = pd.concat([train_data, train_df])
            valid_df = pd.read_csv(
                f'../data/clef2022/CT22_{language}_1A_checkworthy_dev.tsv', sep='\t')
            valid_df['language'] = language
            valid_data = pd.concat([valid_data, valid_df])
            test_df = pd.read_csv(
                f'../data/clef2022/CT22_{language}_1A_checkworthy_dev_test.tsv', sep='\t')
            test_df['language'] = language
            test_data = pd.concat([test_data, test_df])

        train_dataset = HFDataset.from_pandas(train_data)
        valid_dataset = HFDataset.from_pandas(valid_data)
        test_dataset = HFDataset.from_pandas(test_data)
        self.dataset = DatasetDict(
            {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset})

    def preprocess(self, x):
        tweet_text = x['tweet_text']
        # remove urls
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        tweet_text = self._pad_punctuation(tweet_text)
        tweet_text = tweet_text.replace('\n', ' ')
        tweet_text = tweet_text.replace('\t', ' ')
        class_label = self.label_names[x['class_label']]
        return {
            'inputs': f'claim: {tweet_text}',
            'targets': class_label
        }


DATASET_MAPPING = OrderedDict({
    ('clef2022', CLEF2022)
})

TASK_MAPPING = OrderedDict({
    ('check-worthiness', CLEF2022)
})


class DatasetOption:
    @classmethod
    def get(self, dataset):
        if dataset in DATASET_MAPPING:
            return DATASET_MAPPING[dataset]
        raise ValueError(f'Invalid dataset: {dataset}')

    @classmethod
    def get_task(self, task):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task]
        raise ValueError(f'Invalid task: {task}')
