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
from functools import partial

import data.t5.metrics as metrics

Metric = namedtuple('Metric', ['name', 'compute', 'key'])


class Dataset:
    def __init__(self, benchmark_name, subset=None, split=None):
        self.benchmark_name = benchmark_name
        self.subset = subset
        self.split = split
        self.label_names = None
        self.load_dataset()
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }

    def load_dataset(self):
        if self.subset is None:
            self.dataset = hfload_dataset(
                self.benchmark_name)  # , self.split, cache_dir='../../../data/.cache')
        else:
            self.dataset = hfload_dataset(
                self.benchmark_name, self.subset)  # , self.split, cache_dir='../../../data/.cache')

    def _pad_punctuation(self, text):
        """Adds spaces around punctuation."""
        text = regex.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
        # Collapse consecutive whitespace into one space.
        text = regex.sub(r'\s+', ' ', text)
        return text

    def preprocess(self, x):
        return NotImplementedError

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.label_names is not None:
            return max([len(tokenizer.encode(label)) for label in self.label_names])
        return default_max_length

    def postprocess(self, labels, preds, tokenizer, ignore_pad_token_for_loss=True):

        if ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        decoded_preds = [text.strip() for text in decoded_preds]
        decoded_labels = [text.strip() for text in decoded_labels]

        return decoded_labels, decoded_preds

    def tokenize(self, example, tokenizer, max_input_length, max_target_length, padding='max_length', truncation=True):
        inputs = example["inputs"]
        targets = example["targets"]

        model_inputs = tokenizer(inputs, max_length=max_input_length,
                                 padding=padding, truncation=truncation, return_tensors="pt")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length,
                               padding=padding, truncation=truncation, return_tensors="pt")
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels

        return model_inputs

    def postprocess_for_metrics(self, labels, preds):
        # print(labels)
        # print(preds)
        labels = [self.label_names.index(
            label) if label in self.label_names else -1 for label in labels]
        preds = [self.label_names.index(
            pred) if pred in self.label_names else -1 for pred in preds]
        return labels, preds

    def __len__(self):
        return len(self.dataset[self.split_to_data_split[self.split]])


class Record(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='record', split=split)
        self.name = 'record'
        self.label_names = ['Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch']
        self.metrics = [
            Metric(name='Deduplicate metric',
                   compute=metrics.deduplicate_metric(metrics.squad),
                   key=['f1']
                   )
        ]

    def preprocess(self, batch):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["inputs"].extend([inputs] * num_duplicates)
            new_batch["targets"].extend(
                ex["answers"] if num_answers > 0 else ["<unk>"])
            # new_batch["task"].extend([self.name] * num_duplicates)

        return new_batch

    # def preprocess(self, example):
    #     """Convert ReCoRD examples to text2text examples.

    #     For example, a typical example from ReCoRD might look like
    #     {
    #         'passsage': 'This is the passage.',
    #         'query': 'A @placeholder is a bird.',
    #         'entities': ['penguin', 'potato', 'pigeon'],
    #         'answers': ['penguin', 'pigeon'],
    #     }
    #     which this preprocessor would turn into the following two examples:
    #     {
    #         'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
    #                 'potato, pigeon passage: This is the passage.',
    #         'targets': 'penguin',
    #     }
    #     and
    #     {
    #         'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
    #                 'potato, pigeon passage: This is the passage.',
    #         'targets': 'potato',
    #     }

    #     Args:
    #     dataset: a Dataset to process.

    #     Returns:
    #     a Dataset
    #     """
    #     def process_answers(x):
    #         """Helper fn to get one example per answer."""
    #         ex = x.copy()
    #         num_answers = len(ex['answers'])

    #         def duplicate_along_first_dim(t):
    #             n_duplicates = max(num_answers, 1)
    #             return [t] * n_duplicates

    #         for k, v in x.items():
    #             if k != 'idx':
    #                 ex[k] = duplicate_along_first_dim(v)
    #         ex['targets'] = x['answers'] if num_answers > 0 else ['<unk>']
    #         ex['idx'] = {
    #             'passage': duplicate_along_first_dim(x['idx']['passage']),
    #             'query': duplicate_along_first_dim(x['idx']['query']),
    #         }

    #         return ex

    #     def my_fn(x):
    #         """Converts the processed example to text2text strings."""
    #         passage = x['passage']
    #         passage = re.sub(
    #             r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
    #         passage = re.sub(r'\n@highlight\n', '. ', passage)

    #         final_str = f'record query: {x["query"]} entities: {", ".join(x["entities"])} passage: {passage}'
    #         ex = {}

    #         # Store the data index in the returned example (used by eval)
    #         ex['idx/passage'] = x['idx']['passage']
    #         ex['idx/query'] = x['idx']['query']

    #         ex['inputs'] = final_str
    #         # Note that "answers" has been converted to a single string by the
    #         # process_answers function.
    #         ex['targets'] = x['targets']
    #         # Pass-through full list of answers for eval
    #         ex['answers'] = x['answers']
    #         return ex

    #     dataset = self.dataset.map(process_answers)
    #     dataset = dataset.unbatch()
    #     return dataset.map(my_fn)


class STSB(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='stsb', split=split)
        self.name = 'stsb'
        self.label_names = [
            str(np.round(label, decimals=1))
            for label in np.arange(0, 5.1, 0.1)
        ]
        self.metrics = [
            Metric(name='Pearson coefficient',
                   compute=metrics.pearson_corrcoef,
                   key=['pearson_corrcoef']),
            Metric(name='Spearman coefficient',
                   compute=metrics.spearman_corrcoef,
                   key=['spearman_corrcoef'])
        ]

    def preprocess(self, x):
        """
        Convert STSB examples to text2text format.

        For example, a typical example from STSB might look like
        {
            "sentence1": "Three more US soldiers killed in Afghanistan",
            "sentence2": "NATO Soldier Killed in Afghanistan",
            "label": 1.8,
        }

        This example would be transformed to
        {
            "inputs": (
                "stsb sentence1: Three more US soldiers killed in Afghanistan "
                "sentence2: NATO Soldier Killed in Afghanistan"
            ),
            "targets": "1.8",
        }

        Args:
        x: an example to process.
        Returns:
        A preprocessed example.
        """
        text = f'stsb sentence1: {x["sentence1"]} sentence2: {x["sentence1"]}'
        label_string = f'{np.round((x["label"] * 5) / 5, decimals=1)}'
        return {'inputs': text, 'targets': label_string}


class WSC(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='wsc.fixed', split=split)
        self.name = 'wsc'
        self.label_names = self.dataset['train'].features['label'].names
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocess(self, x):
        """Convert WSC examples to text2text format.

        For example, a typical example from WSC might look like
        {
            'text': 'This is a test sentence .',
            'span1_text': 'test',
            'span1_index': 3,
            'span2_text': 'This',
            'span2_index': 0,
            'label': 0
        }

        This example would be transformed to
        {
            'inputs': 'wsc text: # This # is a * test * sentence .',
            'targets': 'False'
        }

        Args:
        x: an example to process.
        Returns:
        A preprocessed example.
        """
        text = x['text']
        text = self._mark_span(text, x['span1_text'], x['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = x['span2_index'] + 2 * \
            int(x['span1_index'] < x['span2_index'])
        text = self._mark_span(text, x['span2_text'], span2_index, '#')

        # Add benchmark name at the start
        final_str = f'wsc text: {text}'

        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {'inputs': final_str, 'targets': label_name}


class MultiRC(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='multirc', split=split)
        self.name = 'multirc'
        self.label_names = self.dataset['train'].features['label'].names
        self.metrics = [
            Metric(name='F1 over all answers',
                   compute=metrics.multirc_f1_over_all_answers,
                   key=['f1']),
            Metric(name='Match all',
                   compute=metrics.mean_group_metric(metrics.all_match),
                   key=['exact_match'])
        ]

    def _remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocess(self, x):
        ex = {}

        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]
        ex['targets'] = label_name
        ex['inputs'] = f'question: {self._remove_markup(x["question"])} answer: {self._remove_markup(x["answer"])} paragraph: {self._remove_markup(x["paragraph"])}'
        return ex


class TriviaQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='trivia_qa', subset='rc', split=split)
        self.name = 'trivia_qa'
        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]

    def _triviaqa_question_answer_context(self, x):
        """Extracts matched contexts and answers.

        Returns all matched (question-context, answer) pairs.

        Args:
          x: A tfds sample.

        Returns:
          Flattened samples: (question-context, answer).
        """
        contexts = []
        if 'entity_pages' in x:
            contexts.append(x['entity_pages']['wiki_context'])
        if 'search_results' in x:
            contexts.append(x['search_results']['search_context'])
        contexts = ''.join(contexts)

        q = self._pad_punctuation(x['question'])
        answers = x['answer']['normalized_aliases']

        combination_size = len(answers) * len(contexts)
        find_answers = []
        selected_answers = []
        join_q_c = []

        for i in range(combination_size):
            context_idx = i // len(answers)
            answer_idx = i % len(answers)

            a = self._pad_punctuation(answers[answer_idx])
            a_ = f'.* {a} .*'
            c = self._pad_punctuation(contexts[context_idx])
            find_a = re.match(a_, c, re.IGNORECASE)

            find_answers.append(find_a)
            selected_answers.append(a)
            join_q_c_str = f'question: {q} context: {c}'
            join_q_c.append(join_q_c_str)

        selected_answers = [a for i, a in enumerate(
            selected_answers) if find_answers[i]]
        selected_join_q_c = [jc for i, jc in enumerate(
            join_q_c) if find_answers[i]]

        return selected_join_q_c, selected_answers

    def preprocess(self):
        def my_fn(x):
            """Create TriviaQA example."""
            join_q_c, a = self._triviaqa_question_answer_context(x)
            return {
                'inputs': join_q_c,
                'targets': a
            }

        dataset = self.dataset.map(my_fn)
        return dataset.unbatch()


class Squad(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='squad', split=split)
        self.name = 'squad'
        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def preprocess(self, x, include_context=True):
        a = self._pad_punctuation(x['answers']['text'][0])
        q = self._pad_punctuation(x['question'])
        c = self._pad_punctuation(x['context'])

        target = a

        if include_context:
            inputs = f'question: {q} context: {c}'
        else:
            inputs = f'squad trivia question: {q}'

        return {
            'inputs': inputs,
            'targets': target,
            'id': x['id'],
            'context': c,
            'question': q,
            'answers': a
        }


class MRQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='mrqa', split=split)
        self.name = 'mrqa'
        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]

    def preprocess(self, x, task_name=None):
        new_ex = {}
        # new_ex['idx'] = x['qid']
        new_ex['question'] = self._pad_punctuation(x['question'])
        new_ex['context'] = self._pad_punctuation(x['context'])
        new_ex['answer'] = self._pad_punctuation(x['answers'][0])
        new_ex['answers'] = self._pad_punctuation(x['answers'])
        strs_to_join = [
            'question:', new_ex['question'], 'context:', new_ex['context']
        ]
        if task_name is not None:
            strs_to_join = [task_name] + strs_to_join
        new_ex['inputs'] = ' '.join(strs_to_join)
        new_ex['targets'] = new_ex['answer']
        return new_ex


class DROP(Dataset):  # need to check in T5 or PromptTuning
    def __init__(self, split='train'):
        super().__init__(benchmark_name='drop', split=split)
        self.name = 'drop'
        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def preprocess(self, x):
        answer = self._pad_punctuation(x['answers_spans']['spans'][0])
        question = self._pad_punctuation(x['question'])
        context = self._pad_punctuation(x['passage'])

        return {
            'inputs': f'question: {question} context: {context}',
            'targets': answer,
        }


class PIQA(Dataset):  # need to check in T5 or PromptTuning
    def __init__(self, split='train'):
        super().__init__(benchmark_name='piqa', split=split)
        self.name = 'piqa'
        self.label_names = self.dataset['train'].features['label'].names
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        return {
            'inputs': f'question: {x["goal"]} choice1: {x["sol1"][0]} choice2: {x["sol2"][0]}',
            'targets': str(x["label"])
        }


class SocialIQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='social_i_qa', split=split)
        self.name = 'social_i_qa'
        self.label_names = ['0', '1', '2']
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def preprocess(self, x):
        return {
            'inputs': f'question: {x["question"]} context: {x["context"]} || choice0: {x["answerA"][0]} || choice1: {x["answerB"][0]} || choice2: {x["answerC"][0]}',
            'targets': str(int(x["label"]) - 1)
        }


class MRPC(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='mrpc', split=split)
        self.name = 'mrpc'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='F1 with invalid',
                   compute=metrics.f1_score_with_invalid, key=['f1']),
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]
        return {
            'inputs': f'sentence1: {x["sentence1"]} sentence2: {x["sentence2"]}',
            'targets': label_name,
        }


class COLA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='cola', split=split)
        self.name = 'cola'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Matthews correlation', compute=metrics.sklearn_metrics_wrapper(
                "matthews_corrcoef", metric_post_process_fn=lambda x: 100 * x), key=['matthews_corrcoef'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentence: {x["sentence"]}',
            'targets': label_name,
        }


class SST2(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='sst2', split=split)
        self.name = 'sst2'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentence: {x["sentence"]}',
            'targets': label_name,
        }


class YelpPolarity(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='yelp_polarity', split=split)
        self.name = 'yelp_polarity'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'test',
            'test': 'test',
        }

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentence: {x["text"]}',
            'targets': label_name,
        }


class QQP(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='qqp', split=split)
        self.name = 'qqp'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='F1 with invalid',
                   compute=metrics.f1_score_with_invalid, key=['f1']),
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'question1: {x["question1"]} question2: {x["question2"]}',
            'targets': label_name,
        }


class MNLI(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='mnli', split=split)
        self.name = 'mnli'
        self.label_names = self.dataset['train'].features['label'].names
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation_mismatched',
            'text': 'test_mismatched',
        }

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class SNLI(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='snli', split=split)
        self.name = 'snli'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class MultiNLI(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='multi_nli', split=split)
        self.name = 'multi_nli'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation_mismatched',
            'test': 'validation_mismatched',
        }

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class DocNLI(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='saattrupdan/doc-nli', split=split)
        self.name = 'doc_nli'
        self.label_names = ['non_entailment', 'entailment']

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'val',
            'test': 'test',
        }

    def preprocess(self, x):
        label_name = x['label']
        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class QNLI(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='qnli', split=split)
        self.name = 'qnli'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'question: {x["question"]} sentence: {x["sentence"]}',
            'targets': label_name,
        }


class RTE(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='rte', split=split)
        self.name = 'rte'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentence1: {x["sentence1"]} sentence2: {x["sentence2"]}',
            'targets': label_name,
        }


class WNLI(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='glue', subset='wnli', split=split)
        self.name = 'wnli'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentence1: {x["sentence1"]} sentence2: {x["sentence2"]}',
            'targets': label_name,
        }


class BoolQ(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='boolq', split=split)
        self.name = 'boolq'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'question: {x["question"]} passage: {x["passage"]}',
            'targets': label_name,
        }


class SuperGLUERTE(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='rte', split=split)
        self.name = 'supeglue_rte'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class CB(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='cb', split=split)
        self.name = 'cb'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='F1 Multiclass', compute=metrics.mean_multiclass_f1(
                num_classes=3), key=['mean_3class_f1']),
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class COPA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='copa', split=split)
        self.name = 'copa'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} choice1: {x["choice1"]} choice2: {x["choice2"]}',
            'targets': label_name,
        }


class WIC(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='super_glue', subset='wic', split=split)
        self.name = 'wic'
        self.label_names = self.dataset['train'].features['label'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentence1: {x["sentence1"]} sentence2: {x["sentence2"]} word: {x["word"]}',
            'targets': label_name,
        }


class WinoGrande(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='winogrande', subset='winogrande_xl', split=split)
        self.name = 'winogrande'
        self.label_names = self.dataset['train'].features['answer'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):

        return {
            'inputs': f'sentence: {x["sentence"]} option0: {x["option0"]} option1: {x["option1"]}',
            'targets': str(int(x["answer"]) - 1),
        }


class ANLI(Dataset):  # Need to resolve version
    def __init__(self, split='train'):
        super().__init__(benchmark_name='anli', split=split)
        self.label_names = self.dataset['train_r1'].features['label'].names
        self.name = 'anli'

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]
        self.split_to_data_split = {
            'train': 'train_r1',
            'validation': 'dev_r1',
            'test': 'test_r1',
        }

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'premise: {x["premise"]} hypothesis: {x["hypothesis"]}',
            'targets': label_name,
        }


class GOEmotions(Dataset):  # Need to resolve som problems with the labels
    def __init__(self, split='train'):
        super().__init__(benchmark_name='go_emotions', subset='simplified', split=split)
        self.name = 'go_emotions'

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def _get_label(self, vector):
        label = np.argmax(vector)
        return str(label)

    def preprocess(self, x):
        return {
            'inputs': f'emotion: {x["text"]}',
            'targets': self._get_label(x['labels']),
        }


class Sentiment140(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='sentiment140', split=split)
        self.name = 'sentiment140'
        # self.label_names = [
        #     str(i)
        #     for i in range(len(self.dataset["train"].features["sentiment"].names))
        # ]
        self.label_names = self.dataset['train'].features['sentiment'].names

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'test',
            'test': 'test',
        }

    def preprocess(self, x):
        label_name = '<unk>' if x['label'] == - \
            1 else self.label_names[x['label']]

        return {
            'inputs': f'sentiment: {x["text"]}',
            'targets': label_name
        }


class SearchQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='search_qa', subset='train_test_val', split=split)
        self.name = 'search_qa'

        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]

    def preprocess(self, x):
        q = self._pad_punctuation(x['question'])

        return {
            'inputs': f'question: {q}',
            'targets': x['answer'],
        }


class HotPotQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='hotpot_qa', subset='fullwiki', split=split)
        self.name = 'hotpot_qa'

        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]

    def preprocess(self, x):
        a = self._pad_punctuation(x['answer'])
        q = self._pad_punctuation(x['question'])
        c = self._pad_punctuation(''.join(x['sentences'][0]))

        inputs = f'question: {q} context: {c}'

        return {
            'inputs': inputs,
            'targets': a[0],
            'id': x['id'],
            'context': c,
            'question': q,
            'answers': a
        }


class NQ_Open(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='nq_open', split=split)
        self.name = 'nq_open'

        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def preprocess(self, x):
        q = self._pad_punctuation(x['question'])

        return {
            'inputs': f'nq question: {q}',
            'targets': x['answer'][0],
            'answers': x['answer']
        }


class CosmosQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='cosmos_qa', split=split)
        self.name = 'cosmos_qa'

        self.metrics = [
            Metric(name='Squad', compute=metrics.squad, key=['f1']),
        ]

    def preprocess(self, x):
        inputs = f'question: {x["question"]} context: {x["context"]} choice0: {x["answer0"]} choice1: {x["answer1"]} choice2: {x["answer2"]} choice3: {x["answer3"][0]}',

        return {
            'inputs': inputs,
            'targets': str(int(x["label"])),
        }


class HellaSwag(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='Rowan/hellaswag', split=split)
        self.name = 'hella_swag'
        self.label_names = ['0', '1', '2', '3']

        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy,
                   key=['accuracy']),
        ]

    def preprocess(self, x):
        label_list = ['0', '1', '2', '3']

        return {
            'inputs': f'context: {x["ctx"]} ending0: {x["endings"][0]} ending1: {x["endings"][1]} ending2: {x["endings"][2]} ending3: {x["endings"][3]}',
            'targets': label_list[x['label']],
        }


class CommonGen(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='common_gen', split=split)
        self.name = 'common_gen'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1']),
        ]

    def preprocess(self, x):
        words = ' '.join(x['concepts'])
        return {
            'inputs': f'generate: {words}',
            'targets': x['target'],
        }


class DART(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='dart', split=split)
        self.name = 'dart'
        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1']),
        ]

    def preprocess(self, x):
        tripleset = '; '.join(x['tripleset'])
        tripleset = re.sub(r'\[(.*?)\]', '', tripleset)
        return {
            'inputs': f'generate: {x["premise"]} {x["hypothesis"]}',
            'targets': x['label'],
        }


class E2ENLG(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='e2e_nlg', split=split)
        self.name = 'e2e_nlg'


class SchemaDialog(Dataset):
    def __init__(self, split='train'):
        super().__init__(subset='schema_guided_dstc8', split=split)
        self.name = 'schema_dialog'


class WebNLG(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='web_nlg_en', split=split)
        self.name = 'web_nlg'

        self.metrics = [
            Metric(name='Bleu', compute=metrics.bleu, key=['bleu'])
        ]

    def preprocess(self, x):

        inputs = f'WebNLG: {x["inputs"][0]}'
        return {
            'inputs': inputs,
            'targets': x['target'],
        }


class WikiAuto(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='wiki_auto_asset', split=split)
        self.name = 'wiki_auto'

        self.metrics = [
            Metric(name='Bleu', compute=metrics.bleu, key=['bleu'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test_asset',
        }

    def preprocess(self, x):
        return {
            'inputs': f'{x["source"]}',
            'targets': x['target'],
        }


class XSUM(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='xsum', split=split)
        self.name = 'xsum'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["document"]}',
            'targets': x["target"],
        }


# wmt14 de-en, wmt15 fr-en, wmt16 ro-en
class WMT(Dataset):
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py
    def __init__(self, benchmark_name, subset, source_language, target_language, split='train'):
        super().__init__(benchmark_name=benchmark_name, subset=subset, split=split)
        self.source_language = source_language
        self.target_language = target_language
        self.name = f'{source_language}_{target_language}'

        self.metrics = [
            Metric(name='Bleu', compute=metrics.bleu, key=['bleu'])
        ]

    def preprocess(self, x):
        """Convert a translation dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'de': 'Das ist gut.', 'en': 'That is good.'}
        If source_language = 'de', target_language = 'en', then the outputs will have
        the format:
            {'inputs': 'translate German to English: Das ist gut.',
            'targets': 'That is good.'}

        Args:
            x: an example to process.
            source_language: source language code (e.g. 'en') to translate from.
            target_language: target language code (e.g. 'de') to translate to.

        Returns:
            A preprocessed example with the format listed above.
        """
        for language in (self.source_language, self.target_language):
            if language != language[:2]:
                raise ValueError(f'Invalid language code: {language}')

        lang_id_to_string = {
            self.source_language: babel.Locale(self.source_language[:2]).english_name,
            self.target_language: babel.Locale(self.target_language[:2]).english_name,
        }

        src_str = 'translate {}'.format(
            lang_id_to_string[self.source_language])
        tgt_str = ' to {}: '.format(lang_id_to_string[self.target_language])

        return {
            'inputs': f'{src_str}{tgt_str}{x[self.source_language]}',
            'targets': x[self.target_language],
        }


class AESLC(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='aeslc', split=split)
        self.name = 'aeslc'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["email_body"]}',
            'targets': x["subject_line"],
        }


class BILLSUM(Dataset):  # Need to check because the dataset have also title
    def __init__(self, split='train'):
        super().__init__(benchmark_name='billsum', split=split)
        self.name = 'billsum'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'test',
            'test': 'ca_test',
        }

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["text"]}',
            'targets': x["summary"],
        }


class GIGAWORD(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gigaword', split=split)
        self.name = 'gigaword'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["document"]}',
            'targets': x["summary"],
        }


class MultiNews(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='multi_news', split=split)
        self.name = 'multi_news'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["document"]}',
            'targets': x["summary"],
        }


class Newsroom(Dataset):  # Datasets also include title of the text
    def __init__(self, split='train'):
        self.benchmark_name = 'newsroom'
        self.split = split
        self.name = 'newsroom'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def load_dataset(self):
        self.dataset = hfload_dataset(self.benchmark_name, '../data/newsroom')

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["text"]}',
            'targets': x["summary"],
        }


class SamSum(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='samsum', split=split)
        self.name = 'samsum'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        """Convert a summarization dataset to a text2text pair.

        For example, say the dataset returns examples of this format:
            {'article': <article>, 'highlights': <summary>}
        If article_key = 'article', summary_key = 'highlights', then the outputs will
        have the format:
            {'inputs': 'summarize': <article>, 'targets': <summary>}

        Args:
            x: an example to process.
        Returns:
            A preprocessed example with the format listed above.
        """
        return {
            'inputs': f'summarize: {x["dialogue"]}',
            'targets': x["summary"],
        }


class C4(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='c4', subset='en', split=split)
        self.name = 'c4'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def preprocess(self, x):
        return {
            'inputs': None,
            'targets': x['text'],
        }


class CNN(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='cnn_dailymail', subset='3.0.0', split=split)
        self.name = 'cnn_dailymail'

        self.metrics = [
            Metric(name='Rouge', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        return {
            'inputs': f'summarize: {x["article"]}',
            'targets': x['highlights'],
        }


class WikiLingua(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='gem', subset='wiki_lingua_english_en', split=split)
        self.name = 'wiki_lingua'
        self.metrics = [
            Metric(name='Rogue', compute=metrics.rouge, key=['rouge1'])
        ]

    def preprocess(self, x):
        source_text = x['source_aligned']['en']
        target_text = x['target_aligned']['en']

        return {
            'inputs': f'{source_text}',
            'targets': target_text,
        }


class CxC(Dataset):
    def __init__(self, split='train', path='../../../data/sts-dataset.csv'):
        self.benchmark_name = 'cxc'
        self.name = 'cxc'
        self.split = split

        self.metrics = [
            Metric(name='Pearson coefficient',
                   compute=metrics.pearson_corrcoef, key=['pearson_corrcoef']),
            Metric(name='Spearman coefficient',
                   compute=metrics.spearman_corrcoef, key=['spearman_corrcoef']),
        ]

        self.load_dataset(path)

    def load_dataset(self, path):
        self.dataset = pd.read_csv(
            f'{path}')

        # dataframe to datasetdict
        self.dataset = DatasetDict({
            'train': HFDataset.from_pandas(self.dataset),
        })

    def preprocess(self, x):
        """
        Convert CxC examples to text2text format.

        For example, a typical example from CxC might look like
        {
            "sentence1": "Three more US soldiers killed in Afghanistan",
            "sentence2": "NATO Soldier Killed in Afghanistan",
            "score": 1.8,
        }

        This example would be transformed to
        {
            "inputs": (
                "sentence1: Three more US soldiers killed in Afghanistan "
                "sentence2: NATO Soldier Killed in Afghanistan"
            ),
            "targets": "1.8",
        }

        Args:
        x: an example to process.
        Returns:
        A preprocessed example.
        """
        text = f'sentence1: {x["sentence1"]} sentence2: {x["sentence1"]}'
        label_string = f'{np.round((x["score"] * 5) / 5, decimals=1)}'
        return {'inputs': text, 'targets': label_string}


class NewsQA(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='legacy107/newsqa', split=split)
        self.name = 'newsqa'
        self.metrics = [
            Metric(name='Rogue', compute=metrics.rouge, key=['rouge1'])
        ]
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def preprocess(self, x, include_context=True):
        a = self._pad_punctuation(x['answers'][0])
        q = self._pad_punctuation(x['question'])
        c = self._pad_punctuation(x['context'])

        if include_context:
            inputs = f'question: {q} context: {c}'
        else:
            inputs = f'question: {q}'

        return {
            'inputs': inputs,
            'targets': a[0],
        }


class Race(Dataset):
    def __init__(self, split='train'):
        super().__init__(benchmark_name='race', subset='middle', split=split)
        self.name = 'race'
        self.label_names = ['0', '1', '2', '3']
        self.metrics = [
            Metric(name='Accuracy', compute=metrics.accuracy, key=['accuracy'])
        ]

    def preprocess(self, x):
        # split options
        A = x['options'][0]
        B = x['options'][1]
        C = x['options'][2]
        D = x['options'][3]
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return {
            'inputs': f'question: {x["question"]} context: {x["article"]} choice0: {A} choice1: {B} choice2: {C} choice3: {D}',
            'targets': str(label2id[x['answer']]),
        }


TASK_MAPPING = OrderedDict([
    ('glue', (COLA, SST2, MRPC, QQP, STSB, MNLI, QNLI, RTE)),
    ('super_glue', (BoolQ, CB, COPA, MultiRC, Record, SuperGLUERTE, WIC, WSC)),
    ('nli', (ANLI, CB, DocNLI, MNLI, QNLI, RTE, SNLI)),
    ('semantic_similarity', (CxC, MRPC, QQP, STSB)),
    ('sentiment_analysis', (GOEmotions, Sentiment140, SST2, YelpPolarity)),
    ('qa', (MRQA, Squad, NewsQA, TriviaQA, SearchQA, HotPotQA, NQ_Open)),
    ('mrqa', (Squad, NewsQA, TriviaQA, SearchQA, HotPotQA, NQ_Open)),
    ('commnonsense_reasoning', (CosmosQA, HellaSwag,
     PIQA, SocialIQA, WinoGrande)),  # alfa-nli is missing
    ('translation', (WMT)),  # three combination of languages
    ('summary', (AESLC, BILLSUM, CNN, WikiLingua,
     GIGAWORD, MultiNews, Newsroom, SamSum, XSUM)),
    ('nlg', (CommonGen, DART, E2ENLG, SchemaDialog, WebNLG, WikiAuto, XSUM, WikiLingua)),
    ('c4', (C4)),
])


class MixtureOfDatasets(Dataset):
    def __init__(self, task, split='train'):
        self.name = f'mixture_{task}'
        self.datasets = DatasetOption.get_task(task)
        self.datasets = [dataset() for dataset in self.datasets]

        self.split = split

        self.metrics = [
            # Metric(name='Accuracy', compute=metrics.accuracy,
            #        key=['accuracy']),
        ]

    def get_max_target_length(self, tokenizer, default_max_length):
        print(
            [dataset.get_max_target_length(tokenizer, default_max_length)
             for dataset in self.datasets]
        )
        return max([
            dataset.get_max_target_length(tokenizer, default_max_length)
            for dataset in self.datasets
        ])


# create mapping for above datasets
DATASET_MAPPING = OrderedDict([
    ('record', Record),
    ('stsb', STSB),
    ('wsc', WSC),
    ('multi_rc', MultiRC),
    ('trivia_qa', TriviaQA),
    ('squad', Squad),
    ('mrqa', MRQA),
    ('drop', DROP),
    ('piqa', PIQA),
    ('social_i_qa', SocialIQA),
    ('mrpc', MRPC),
    ('cola', COLA),
    ('sst2', SST2),
    ('yelp_polarity', YelpPolarity),
    ('qqp', QQP),
    ('mnli', MNLI),
    ('snli', SNLI),
    ('multi_nli', MultiNLI),
    ('doc_nli', DocNLI),
    ('qnli', QNLI),
    ('rte', RTE),
    ('wnli', WNLI),
    ('boolq', BoolQ),
    ('superglue-rte', SuperGLUERTE),
    ('cb', CB),
    ('copa', COPA),
    ('wic', WIC),
    ('winogrande', WinoGrande),
    ('anli', ANLI),
    ('go_emotions', GOEmotions),
    ('sentiment140', Sentiment140),
    ('search_qa', SearchQA),
    ('hotpot_qa', HotPotQA),
    ('nq_open', NQ_Open),
    ('cosmos_qa', CosmosQA),
    ('hellaswag', HellaSwag),
    ('common_gen', CommonGen),
    ('dart', DART),
    ('e2e_nlg', E2ENLG),
    ('schema_guided_dialog', SchemaDialog),
    ('web_nlg_en', WebNLG),
    ('wiki_auto_asset', WikiAuto),
    ('xsum', XSUM),
    ('wmt14', WMT),
    ('wmt15', WMT),
    ('wmt16', WMT),
    ('aeslc', AESLC),
    ('billsum', BILLSUM),
    ('gigaword', GIGAWORD),
    ('multi_news', MultiNews),
    ('newsroom', Newsroom),
    ('samsum', SamSum),
    ('c4', C4),
    ('cnn_dailymail', CNN),
    ('wiki_lingua', WikiLingua),
    ('cxc', CxC),
    ('newsqa', NewsQA),
    ('race', Race),
    ('mixture', MixtureOfDatasets),
])


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
