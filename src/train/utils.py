import yaml
from transformers import TrainingArguments
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW, AdamWeightDecay


def create_arguments(num_samples, config, metrics):
    batch_size = config.batch_size
    model_name = config.model_name
    lr = config.learning_rate
    eval_strategy = config.eval_strategy
    gradient_accumulation_steps = config.gradient_steps
    epochs = config.epochs
    wd = config.weight_decay
    metrics = metrics
    use_fp16 = config.fp16

    logging_steps = len(num_samples) // (batch_size * epochs)
    # eval around each 2000 samples
    logging_steps = round(2000 / (batch_size * gradient_accumulation_steps))

    return TrainingArguments(
        output_dir=f"{config.output_path}{model_name}-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy=eval_strategy,
        logging_steps=logging_steps,
        save_strategy=eval_strategy,
        save_steps=logging_steps,
        eval_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        weight_decay=wd,
        push_to_hub=False,
        report_to="wandb",
        metric_for_best_model=metrics,
        fp16=use_fp16,  # mdeberta not working with fp16
        warmup_steps=0,
        logging_dir=f"{config.output_path}{model_name}-finetuned/logs",

    )


def get_optimizer(config, model):

    optimizer = config.optimizer
    warm_init = config.warm_init
    optimizer_lr = config.learning_rate

    if optimizer == 'AdafactorSchedule':
        return AdafactorSchedule(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=warm_init,
            lr=optimizer_lr
        )
    elif optimizer == 'AdamW':
        return AdamW(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=warm_init,
            lr=optimizer_lr
        )
    elif optimizer == 'AdamWeightDecay':
        return AdamWeightDecay(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=warm_init,
            lr=optimizer_lr
        )

    return Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=warm_init,
        lr=optimizer_lr
    )


def get_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
