import os
import asdict
from huggingface_hub import hf_hub_download
import json
import inspect

CONFIG_NAME = 'adapter_config.json'

class PromptTuningConfig:
    def __init__(
        self,
        auto_mapping=None, 
        base_model_or_path=None,
        revision=None,
        task_type='SEQ_2_SEQ_LM',
        inference_mode=False,
        num_virtual_tokens=20,
        token_dim=None, 
        num_transformer_submodules=None, 
        num_layers=None, 
        init_type='random', 
        init_text=None, 
        num_attention_heads=None, 
        tokenizer_name_or_path='t5-base'
    ):
        self.auto_mapping = auto_mapping
        self.base_model_or_path = base_model_or_path
        self.revision = revision
        self.task_type = task_type
        self.inference_mode = inference_mode
        self.num_virtual_tokens = num_virtual_tokens
        self.token_dim = token_dim
        self.num_transformer_submodules = num_transformer_submodules
        self.num_layers = num_layers
        self.init_type = init_type
        self.init_text = init_text
        self.num_attention_heads = num_attention_heads
        self.tokenizer_name_or_path = tokenizer_name_or_path


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = pretrained_model_name_or_path

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            raise ValueError("Can't find a configuration file in the specified directory")

        loaded_attributes = cls.from_json_file(config_file)
        config_cls = cls

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        kwargs = {**class_kwargs, **loaded_attributes}
        return config_cls(**kwargs)
        
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        
        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop('auto_mapping_dict', None)

        output_dict = asdict(self)
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)
        
        output_path = os.path.join(save_directory, CONFIG_NAME)
        if auto_mapping_dict is not None:
            output_dict['auto_mapping_dict'] = auto_mapping_dict

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(output_dict, ensure_ascii=False, indent=2, sort_keys=True))


    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

        