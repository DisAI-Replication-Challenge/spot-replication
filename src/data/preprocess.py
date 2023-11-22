import functools
import logging

logging.basicConfig(level=logging.INFO)


def preprocess_data(dataloader, tokenizer, max_input_length=512, max_target_length=512, padding='max_length', truncation=True, test_set=False):

    func = functools.partial(
        dataloader.tokenize,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        padding=padding,
        truncation=truncation,
    )

    remove_columns = dataloader.dataset['train'].column_names

    if test_set:
        logging.info("Preprocessing test data")
        test_data = dataloader.dataset['test'].map(dataloader.preprocess)

        logging.info("Tokenizing test data")
        test_data_tokenized = test_data.map(
            func, batched=True, remove_columns=remove_columns)

        return None, None, test_data_tokenized
    else:
        logging.info("Preprocessing train data")
        train_data = dataloader.dataset['train'].map(dataloader.preprocess)
        logging.info("Tokenizing train data")
        train_data_tokenized = train_data.map(
            func, batched=True, remove_columns=remove_columns)

        # find whether validation split exists in dataloader.dataset
        valid_data_tokenized = None
        if 'validation' in dataloader.dataset:
            logging.info("Preprocessing validation data")
            valid_data = dataloader.dataset['validation'].map(
                dataloader.preprocess)
            logging.info("Tokenizing validation data")
            valid_data_tokenized = valid_data.map(
                func, batched=True, remove_columns=remove_columns)

        return train_data_tokenized, valid_data_tokenized, None
