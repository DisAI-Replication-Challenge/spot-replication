from datasets import load_dataset as hfload_dataset
import numpy as np
import pandas as pd
import os
import re
from datasets import DatasetDict
from datasets import Dataset as HFDataset
from collections import OrderedDict, namedtuple
import numpy as np

import data.metrics as metrics
from data.dataset import Dataset

Metric = namedtuple('Metric', ['name', 'compute', 'key'])


def convert_language(language):
    if language == 'english':
        return 'en'
    elif language == 'spanish':
        return 'es'
    elif language == 'turkish':
        return 'tr'
    elif language == 'dutch':
        return 'nl'
    elif language == 'bulgarian':
        return 'bg'
    elif language == 'arabic':
        return 'ar'
    elif language == 'czech':
        return 'cs'
    elif language == 'hungarian':
        return 'hu'
    elif language == 'polish':
        return 'pl'
    elif language == 'slovak':
        return 'sk'
    else:
        raise ValueError(f'Invalid language: {language}')


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


class CLEF2022KInIT(Dataset):
    def __init__(self, split='train', language='english'):
        self.languages = [convert_language(language)]
        super().__init__(None, None, split)
        self.name = 'CLEF2022KInIT'
        self.label_names = ['unworthy', 'checkworthy']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def load_dataset(self):
        dataset = hfload_dataset(
            "ivykopal/fact-checking-datasets", data_dir='clef2022', token=os.environ['HF_API_KEY'])
        # get specific language
        self.dataset = dataset.filter(
            lambda x: x['lang'] in self.languages)

    def preprocess(self, x):
        text = x['text']
        # remove urls
        text = re.sub(r'http\S+', '', text)
        text = self._pad_punctuation(text)
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        class_label = self.label_names[x['claim']]
        return {
            'inputs': f'claim: {text}',
            'targets': class_label
        }


class CLEF2023KInIT(Dataset):
    def __init__(self, split='train', language='english'):
        self.languages = [convert_language(language)]
        super().__init__(None, None, split)
        self.name = 'CLEF2023KInIT'
        # self.languages = ["arabic", "bulgarian",
        #                   "dutch", "english", "spanish", "turkish"]
        self.label_names = ['unworthy', 'checkworthy']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def load_dataset(self):
        dataset = hfload_dataset(
            "ivykopal/fact-checking-datasets", data_dir='clef2023', token=os.environ['HF_API_KEY'])
        # get specific language
        self.dataset = dataset.filter(
            lambda x: x['lang'] in self.languages)

    def preprocess(self, x):
        text = x['text']
        # remove urls
        text = re.sub(r'http\S+', '', text)
        text = self._pad_punctuation(text)
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        class_label = self.label_names[x['claim']]
        return {
            'inputs': f'claim: {text}',
            'targets': class_label
        }


class MONANT(Dataset):
    def __init__(self, split='train', language='english'):
        self.language = convert_language(language)
        super().__init__(None, None, split)
        self.name = 'monant'
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
        if os.path.exists('../data/monant/train.csv'):
            train = pd.read_csv('../data/monant/train.csv')
            dev = pd.read_csv('../data/monant/dev.csv')
            test = pd.read_csv('../data/monant/test.csv')
        else:
            monat_data = pd.read_csv(
                f'../data/monant/monant.csv', sep=';')
            synthetic_data = pd.read_csv(
                f'../data/monant/synthetic.csv')

            datasets = pd.concat([monat_data, synthetic_data])
            datasets = datasets.sample(frac=1).reset_index(drop=True)
            split_ratio = [0.7, 0.15]
            idx = [int(datasets.shape[0] * split_ratio[i]) for i in range(2)]

            train = datasets.iloc[:idx[0]].copy()
            dev = datasets.iloc[idx[0]: idx[0] + idx[1]].copy()
            test = datasets.iloc[idx[0] + idx[1]:, :].copy()

            train.to_csv('../data/monant/train.csv', index=False)
            dev.to_csv('../data/monant/dev.csv', index=False)
            test.to_csv('../data/monant/test.csv', index=False)

        train['claim'] = train['claim'].astype(int)
        dev['claim'] = dev['claim'].astype(int)
        test['claim'] = test['claim'].astype(int)

        # get based on language
        train = train[[self.language, 'claim']]
        dev = dev[[self.language, 'claim']]
        test = test[[self.language, 'claim']]

        self.dataset = DatasetDict({
            'train': HFDataset.from_pandas(train),
            'valid': HFDataset.from_pandas(dev),
            'test': HFDataset.from_pandas(test)
        })

    def preprocess(self, x):
        text = x[self.language]
        # remove urls
        text = re.sub(r'http\S+', '', text)
        text = self._pad_punctuation(text)
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        class_label = self.label_names[x['claim']]

        return {
            'inputs': f'claim: {text}',
            'targets': class_label
        }


class CLEF2021(Dataset):
    def __init__(self, split='train', languages='english'):
        self.languages = [languages]
        super().__init__(None, None, split)
        self.name = 'CLEF2021'
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
                f'../data/clef2021/dataset_train_v1_{language}.tsv', sep='\t')
            train_df['language'] = language
            train_data = pd.concat([train_data, train_df])
            valid_df = pd.read_csv(
                f'../data/clef2021/dataset_dev_v1_{language}.tsv', sep='\t')
            valid_df['language'] = language
            valid_data = pd.concat([valid_data, valid_df])
            test_df = pd.read_csv(
                f'../data/clef2021/dataset_test_{language}.tsv', sep='\t')
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
        class_label = self.label_names[x['check_worthiness']]
        return {
            'inputs': f'claim: {tweet_text}',
            'targets': class_label
        }


class CLEF2023(Dataset):
    def __init__(self, split='train', languages='english'):
        self.languages = [languages]
        super().__init__(None, None, split)
        self.name = 'CLEF2021'
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
                f'../data/clef2023/CT23_1B_checkworthy_{language}_train.tsv', sep='\t')
            train_df['language'] = language
            train_data = pd.concat([train_data, train_df])
            valid_df = pd.read_csv(
                f'../data/clef2023/CT23_1B_checkworthy_{language}_dev.tsv', sep='\t')
            valid_df['language'] = language
            valid_data = pd.concat([valid_data, valid_df])
            test_df = pd.read_csv(
                f'../data/clef2023/CT23_1B_checkworthy_{language}_test_gold.tsv', sep='\t')
            test_df['language'] = language
            test_data = pd.concat([test_data, test_df])

        train_dataset = HFDataset.from_pandas(train_data)
        valid_dataset = HFDataset.from_pandas(valid_data)
        test_dataset = HFDataset.from_pandas(test_data)
        self.dataset = DatasetDict(
            {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset})

    def preprocess(self, x):
        def convert_label(label):
            if label.lower() == 'no':
                return 0
            elif label.lower() == 'yes':
                return 1
            else:
                raise ValueError(f'Invalid label: {label}')

        tweet_text = x['Text']
        # remove urls
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        tweet_text = self._pad_punctuation(tweet_text)
        tweet_text = tweet_text.replace('\n', ' ')
        tweet_text = tweet_text.replace('\t', ' ')
        class_label = self.label_names[convert_label(x['class_label'])]
        return {
            'inputs': f'claim: {tweet_text}',
            'targets': class_label
        }


class LESA2021(Dataset):
    def __init__(self, split='train', language='english'):
        self.language = convert_language(language)
        super().__init__(None, None, split)
        self.name = 'LESA2021'
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
        if not os.path.exists('../data/lesa2021/train.csv'):
            files = ['noisy', 'semi', 'struct']
            datasets = pd.DataFrame()
            for file in files:
                datasets = pd.concat([datasets, pd.read_csv(
                    f'../data/lesa2021/{file}.csv')])

            split_ratio = [0.7, 0.15]
            idx = [int(datasets.shape[0] * split_ratio[i]) for i in range(2)]

            train = datasets.iloc[:idx[0]].copy()
            dev = datasets.iloc[idx[0]: idx[0] + idx[1]].copy()
            test = datasets.iloc[idx[0] + idx[1]:, :].copy()

            train.to_csv('../data/lesa2021/train.csv', index=False)
            dev.to_csv('../data/lesa2021/dev.csv', index=False)
            test.to_csv('../data/lesa2021/test.csv', index=False)
        else:
            train = pd.read_csv('../data/lesa2021/train.csv')
            dev = pd.read_csv('../data/lesa2021/dev.csv')
            test = pd.read_csv('../data/lesa2021/test.csv')

        # get based on language
        # convert caim to int
        train['claim'] = train['claim'].astype(int)
        dev['claim'] = dev['claim'].astype(int)
        test['claim'] = test['claim'].astype(int)

        train = train[[self.language, 'claim']]
        dev = dev[[self.language, 'claim']]
        test = test[[self.language, 'claim']]

        self.dataset = DatasetDict({
            'train': HFDataset.from_pandas(train),
            'valid': HFDataset.from_pandas(dev),
            'test': HFDataset.from_pandas(test)
        })

    def preprocess(self, x):
        text = x[self.language]
        # remove urls
        text = re.sub(r'http\S+', '', text)
        text = self._pad_punctuation(text)
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        class_label = self.label_names[x['claim']]
        return {
            'inputs': f'claim: {text}',
            'targets': class_label
        }


class LIAR(Dataset):
    def __init__(self, split='train', multiclass=False):
        super().__init__('liar', None, split)
        self.multiclass = multiclass
        self.name = 'liar'
        if multiclass:
            self.label_names = ['false', 'half-true', 'mostly-true',
                                'true', 'barely-true', 'pants-fire']
        else:
            self.label_names = ['false', 'true']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        text = x['statement']
        # remove urls
        text = re.sub(r'http\S+', '', text)
        text = self._pad_punctuation(text)
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        if self.multiclass:
            class_label = self.label_names[x['label']]
        else:
            if x['label'] in [0, 1, 5]:
                class_label = self.label_names[0]
            else:
                class_label = self.label_names[1]

        return {
            'inputs': f'claim: {text}',
            'targets': class_label
        }


class XFact(Dataset):
    def __init__(self, split='train', language='english', use_evidence=False):
        self.language = convert_language(language)
        super().__init__('xfact', None, split)
        self.use_evidence = use_evidence
        self.name = 'xfact'
        self.label_names = ['false', 'true']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def load_dataset(self):
        train_df = pd.read_csv('../data/xfact/train.all.tsv',
                               sep='\t', encoding='utf-8', on_bad_lines='skip')
        valid_df = pd.read_csv('../data/xfact/dev.all.tsv',
                               sep='\t', encoding='utf-8', on_bad_lines='skip')
        test_df = pd.read_csv('../data/xfact/test.all.tsv',
                              sep='\t', encoding='utf-8', on_bad_lines='skip')

        def convert_label(label):
            if label in ['false', 'partly true/misleading']:
                return 0
            elif label == 'true':
                return 1
            else:
                return np.nan

        train_df['label'] = train_df['label'].apply(convert_label)
        valid_df['label'] = valid_df['label'].apply(convert_label)
        test_df['label'] = test_df['label'].apply(convert_label)

        # remove rows with nan in label
        train_df = train_df.dropna(subset=['label'])
        valid_df = valid_df.dropna(subset=['label'])
        test_df = test_df.dropna(subset=['label'])

        # get based on language
        train_df = train_df[train_df['language'] == self.language]
        valid_df = valid_df[valid_df['language'] == self.language]
        test_df = test_df[test_df['language'] == self.language]

        self.dataset = DatasetDict({
            'train': HFDataset.from_pandas(train_df),
            'valid': HFDataset.from_pandas(valid_df),
            'test': HFDataset.from_pandas(test_df)
        })

    def preprocess(self, x):
        text = x['claim']
        # remove urls
        text = re.sub(r'http\S+', '', text)
        text = self._pad_punctuation(text)
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        class_label = self.label_names[x['label']]

        if self.use_evidence:
            evidences = [x[f'evidence_{idx}'] for idx in range(1, 6)]
            evidences = [
                self._pad_punctuation(
                    re.sub(r'http\S+', '', evidence)).replace('\t', ' ').replace('\n', ' ')
                for evidence in evidences
            ]
            evidences = [
                f'evidence{idx + 1}: {evidence}'
                for idx, evidence in enumerate(evidences)
            ]

            evidence_text = ' '.join(evidences)
            input_text = f'claim: {text} {evidence_text}'
        else:
            input_text = f'claim: {text}'

        return {
            'inputs': input_text,
            'targets': class_label
        }


class FEVER(Dataset):
    def __init__(self, split='train', language='english'):
        self.language = convert_language(language)
        super().__init__('fever', 'v1.0', split)
        self.name = 'fever'
        self.label_names = ['supports', 'refutes', 'not enough info']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }

    def preprocess(self, x):
        pass


class CSFEVER(Dataset):
    def __init__(self, split='train', language='english'):
        self.language = convert_language(language)
        super().__init__('ctu-aic/csfever', None, split)
        self.name = 'csfever'
        self.label_names = ['supports', 'refutes', 'not enough info']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }

    def preprocess(self, x):
        pass


class CTKFACTS(Dataset):
    def __init__(self, split='train', language='english'):
        self.language = convert_language(language)
        super().__init__('ctu-aic/ctkfacts', None, split)
        self.name = 'ctkfacts'
        self.label_names = ['supports', 'refutes', 'not enough info']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }

    def preprocess(self, x):
        pass


DATASET_MAPPING = OrderedDict({
    ('clef2022', CLEF2022),
    ('clef2022kinit', CLEF2022KInIT),
    ('clef2023kinit', CLEF2023KInIT),
    ('monant', MONANT),
    ('clef2021', CLEF2021),
    ('clef2023', CLEF2023),
    ('lesa2021', LESA2021),
    ('liar', LIAR),
    ('xfact', XFact),
    ('fever', FEVER),
})

TASK_MAPPING = OrderedDict({
    ('check-worthiness', (CLEF2022, CLEF2022KInIT,
     CLEF2023KInIT, MONANT, CLEF2021, CLEF2023, LESA2021)),
    ('fake-news-detection', (LIAR, XFact)),
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
