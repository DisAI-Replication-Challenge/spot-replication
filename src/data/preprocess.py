import functools


def preprocess_data(dataloader, tokenizer, max_input_length=512, max_target_length=512, padding='max_length', truncation=True):
    train_data = dataloader.dataset['train'].map(dataloader.preprocess)
    valid_data = dataloader.dataset['validation'].map(dataloader.preprocess)
    test_data = dataloader.dataset['test'].map(dataloader.preprocess)

    func = functools.partial(
        dataloader.tokenize,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        padding=padding,
        truncation=truncation,
    )

    remove_columns = dataloader.dataset['train'].column_names

    train_data_tokenized = train_data.map(
        func, batched=True, remove_columns=remove_columns)
    valid_data_tokenized = valid_data.map(
        func, batched=True, remove_columns=remove_columns)
    test_data_tokenized = test_data.map(
        func, batched=True, remove_columns=remove_columns)

    return train_data_tokenized, valid_data_tokenized, test_data_tokenized
