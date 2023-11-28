from .utils import _prepare_prompt_learning_config
from .prompt_tuning import PromptTuningForSeq2SeqLM


def get_prompt_tuning_model(model, peft_config, adapter_name='default'):
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config = _prepare_prompt_learning_config(peft_config, model_config)

    return PromptTuningForSeq2SeqLM(model, peft_config, adapter_name=adapter_name)
