import functools
import logging

logging.basicConfig(level=logging.INFO)


def preprocess_data(
    dataloader,
    tokenizer,
    max_input_length=128,
    max_target_length=None,
    padding='max_length',
    truncation=True,
    test_set=False
):
    if max_target_length is None:
        max_target_length = dataloader.get_max_target_length(
            tokenizer, max_target_length)
        print(max_target_length)

    func = functools.partial(
        dataloader.tokenize,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        padding=padding,
        truncation=truncation,
    )

    key = dataloader.split_to_data_split['train']
    remove_columns = dataloader.dataset[key].column_names

    if test_set:
        logging.info("Preprocessing test data")
        key = dataloader.split_to_data_split['test']
        test_data = dataloader.dataset[key].map(
            dataloader.preprocess, batched=True if dataloader.name == 'record' else False,
            remove_columns=remove_columns, load_from_cache_file=False)

        logging.info("Tokenizing test data")
        test_data_tokenized = test_data.map(
            func, batched=True, load_from_cache_file=False)

        return None, None, test_data_tokenized
    else:
        logging.info("Preprocessing train data")
        key = dataloader.split_to_data_split['train']
        train_data = dataloader.dataset[key].map(
            dataloader.preprocess, batched=True if dataloader.name == 'record' else False,
            remove_columns=remove_columns, load_from_cache_file=False)
        logging.info("Tokenizing train data")
        train_data_tokenized = train_data.map(
            func, batched=True, load_from_cache_file=False)

        key = dataloader.split_to_data_split['validation']
        logging.info("Preprocessing validation data")
        valid_data = dataloader.dataset[key].map(
            dataloader.preprocess, batched=True if dataloader.name == 'record' else False,
            remove_columns=remove_columns, load_from_cache_file=False)
        logging.info("Tokenizing validation data")
        valid_data_tokenized = valid_data.map(
            func, batched=True, load_from_cache_file=False)
        # elif 'test' in dataloader.dataset:
        #     logging.info("Preprocessing test data")
        #     valid_data = dataloader.dataset['test'].map(
        #         dataloader.preprocess)
        #     logging.info("Tokenizing test data")
        #     valid_data_tokenized = valid_data.map(
        #         func, batched=True, remove_columns=remove_columns)

        print(train_data_tokenized)
        print(valid_data_tokenized)
        return train_data_tokenized, valid_data_tokenized, None
