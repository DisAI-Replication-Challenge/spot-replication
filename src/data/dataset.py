from datasets import load_dataset as hfload_dataset
import regex
import numpy as np


class Dataset:
    def __init__(self, benchmark_name, subset=None, split=None):
        self.benchmark_name = benchmark_name
        self.subset = subset
        self.split = split
        self.label_names = None
        # if benchmark_name is not None:
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
        labels = [self.label_names.index(
            label) if label in self.label_names else -1 for label in labels]
        preds = [self.label_names.index(
            pred) if pred in self.label_names else -1 for pred in preds]
        return labels, preds

    def __len__(self):
        return len(self.dataset[self.split_to_data_split[self.split]])
