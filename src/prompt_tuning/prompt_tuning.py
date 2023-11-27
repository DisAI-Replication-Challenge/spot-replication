from src.prompt_tuning.config import PromptTuningConfig
import torch
import math
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel
from copy import deepcopy
from safetensors.torch import storage_ptr, storage_size
import inspect
from huggingface_hub import hf_hub_download


WEIGHTS_NAME = 'adapter_model.bin'


class PromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        init_type = config.init_type

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embeddings = torch.nn.Embedding(total_virtual_tokens)
        if init_type == 'text':
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            init_text = config.init_text
            init_tokens = tokenizer(init_text)['input_ids']
            num_tokens = len(init_tokens)
            if num_tokens > total_virtual_tokens:
                init_tokens = init_tokens[:total_virtual_tokens]
            elif num_tokens < total_virtual_tokens:
                init_tokens += [tokenizer.pad_token_id] * math.ceil(total_virtual_tokens / num_tokens)

            init_tokens = init_tokens[:total_virtual_tokens]
            word_embedding_weights = word_embeddings(torch.LongTensor(init_tokens)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)

    def forward(self, indices):
        return self.embeddings(indices)

def get_prompt_tuning_model(model, config, adapter_name='default'):
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    return PromptTuningForSeq2SeqLM(model, config, adapter_name=adapter_name)



class PromptTuningForSeq2SeqLM:
    def __init__(self, model, config, adapter_name='default'):
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.peft_type = 'prompt_tuning'
        self.adapter_name = adapter_name

        self._peft_config = {adapter_name: config}

        self.base_model = model
        self.add_adapter(adapter_name, config)
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))

        selected_adapters = list(self._peft_config.keys())
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            prompt_config = self._peft_config[adapter_name]
            output_state_dict = self.get_peft_model_state_dict(state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name)

            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            if prompt_config.base_model_or_path is None:
                prompt_config.base_mode_name_or_path = (
                    self.base_model.__dict__.get('name_or_path', None)
                )
            
            inference_mode = prompt_config.inference_mode
            prompt_config.inference_mode = True
            auto_mapping_dict = None
            prompt_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            prompt_config.inference_mode = inference_mode
          
    
    def get_peft_model_state_dict(self, state_dict=None, adapter_name='default'):
        config = self._peft_config[adapter_name]
        if state_dict is None:
            state_dict = self.state_dict()

        to_return = {}
        if config.inferece_mode:
            prompt_embeddings = self.prompt_encoder[adapter_name].embeddings.weight
        else:
            prompt_embeddings = self._get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
        to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
        return to_return


    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            inputs_embeds=None, 
            decoder_input_ids=None, 
            decoder_attention_mask=None, 
            decoder_inputs_embeds=None, 
            labels=None,
            output_attentions=None,
            output_hidden_states=None, 
            return_dict=None,
            **kwargs
        ):
        prompt_config = self._peft_config[self.adapter_name]
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if decoder_attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(decoder_attention_mask.device)

        kwargs.update({
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        })

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(
                attention_mask.device
            )
            kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts[:,:, prompt_config.num_virtual_tokens], inputs_embeds), dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs
        )

    def generate(self, **kwargs):
        prompt_config = self._peft_config[self.adapter_name]
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            kwargs['position_ids'] = None
            kwargs['token_type_ids'] = None
            kwargs = deepcopy(kwargs)

            if 'encoder_outputs' in kwargs:
                del kwargs['encoder_outputs']
            
            inputs_ids = kwargs.pop('input_ids', None)
            inputs_embeds = self.word_embeddings(inputs_ids)
            batch_size = inputs_embeds.shape[0]
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)

            inputs_embeds = torch.cat((prompts[:,:, prompt_config.num_virtual_tokens], inputs_embeds), dim=1)
            kwargs['inouts_embeds'] = inputs_embeds

            if 'attention_mask' in kwargs:
                prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(
                    kwargs['attention_mask'].device
                )
                kwargs['attention_mask'] = torch.cat((prefix_attention_mask, kwargs['attention_mask']), dim=1)

            return self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name='default', is_trainable=False, config=None, **kwargs):
        config = PromptTuningConfig.from_pretrained(model_id, **kwargs)

        model = cls(model, config, adapter_name=adapter_name)
        model.load_adapter(model_id, adapter_name=adapter_name, is_trainable=is_trainable, **kwargs)
        return model


    def _setup_prompt_encoder(self, adapter_name):
        config = self._peft_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2
        
        for named_param, value in list(transformer_backbone.named_parameters()):
            deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()
        
    def _get_prompt_embedding_to_save(self, adapter_name):
        pass
    
    def get_prompt(self, batch_size):
        prompt_config = self._peft_config[self.adapter_name]
        prompt_encoder = self.prompt_encoder[self.adapter_name]
        prompt_tokens = (
            self.prompt_tokens[self.adapter_name]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embeddings.weight.device)
        )
        if prompt_config.inferece_mode:
            return prompt_encoder.embeddings.weight.repeat(batch_size, 1, 1)
        else:
            return prompt_encoder(prompt_tokens)

    @classmethod
    def _split_kwargs(cls, kwargs):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def load_adapter(self, model_id, adapter_name='default', is_trainable=False, **kwargs): # TODO: I ended here
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()

        adapter_weights = load_adapter_weights(model_id, device=torch_device)
        # load_result = 
        pass


    def add_adapter(self, adapter_name, prompt_config):
        # self.peft_config[adapter_name] = config
        if hasattr(self.config, "to_dict"):
            dict_config = self.config.to_dict()
        else:
            dict_config = self.config

        prompt_config = self._prepare_prompt_learning_config(prompt_config, dict_config)
        self._setup_prompt_encoder(adapter_name)
    
    def _prepare_prompt_learning_config(prompt_config, model_config):
        if prompt_config.num_layers is None:
            if "num_hidden_layers" in model_config:
                num_layers = model_config["num_hidden_layers"]
            elif "num_layers" in model_config:
                num_layers = model_config["num_layers"]
            elif "n_layer" in model_config:
                num_layers = model_config["n_layer"]
            else:
                raise ValueError("Please specify `num_layers` in `prompt_config`")
            prompt_config.num_layers = num_layers

        if prompt_config.token_dim is None:
            if "hidden_size" in model_config:
                token_dim = model_config["hidden_size"]
            elif "n_embd" in model_config:
                token_dim = model_config["n_embd"]
            elif "d_model" in model_config:
                token_dim = model_config["d_model"]
            else:
                raise ValueError("Please specify `token_dim` in `peft_config`")
            prompt_config.token_dim = token_dim

        if prompt_config.num_attention_heads is None:
            if "num_attention_heads" in model_config:
                num_attention_heads = model_config["num_attention_heads"]
            elif "n_head" in model_config:
                num_attention_heads = model_config["n_head"]
            elif "num_heads" in model_config:
                num_attention_heads = model_config["num_heads"]
            elif "encoder_attention_heads" in model_config:
                num_attention_heads = model_config["encoder_attention_heads"]
            else:
                raise ValueError("Please specify `num_attention_heads` in `peft_config`")
            prompt_config.num_attention_heads = num_attention_heads

        if getattr(prompt_config, "encoder_hidden_size", None) is None:
            setattr(prompt_config, "encoder_hidden_size", prompt_config.token_dim)

        return prompt_config
    

def _get_batch_size(input_ids, inputs_embeds):
    if input_ids is not None:
        return input_ids.shape[0]
    elif inputs_embeds is not None:
        return inputs_embeds.shape[0]
    else:
        return None


def id_tensor_storage(tensor):
    unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)


def infer_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_adapter_weights(model_id, device=None):
    if device is None:
        device = infer_device()
    
    adapter_weights = torch.load(os.path.join(model_id, WEIGHTS_NAME), map_location=device)
    return adapter_weights