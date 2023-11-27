from config import PromptTuningConfig
import torch
import os
from transformers import PreTrainedModel
from copy import deepcopy
from safetensors.torch import storage_ptr, storage_size
import inspect
from huggingface_hub import hf_hub_download
from model import PromptEmbedding
from utils import _prepare_prompt_learning_config, _get_batch_size, get_peft_model_state_dict, infer_device, load_adapter_weights, set_peft_model_state_dict
from transformers.utils import PushToHubMixin
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules


WEIGHTS_NAME = 'adapter_model.bin'


class PromptTuningForSeq2SeqLM(PushToHubMixin, torch.nn.Module):

    # DONE
    def __init__(self, model, peft_config, adapter_name='default'):
        super().__init__()
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.peft_type = 'prompt_tuning'
        self.adapter_name = adapter_name
        self.device = model.device

        self._peft_config = {adapter_name: peft_config}
        self.base_model = model
        self.config = getattr(self.base_model, "config",
                              {"model_type": "custom"})
        self.add_adapter(adapter_name, peft_config)

        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    # DONE
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError(
                "Provided path ({}) should be a directory, not a file".format(save_directory))

        selected_adapters = list(self._peft_config.keys())
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            prompt_config = self._peft_config[adapter_name]
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name)

            output_dir = os.path.join(
                save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(
                output_dir, WEIGHTS_NAME))

            if prompt_config.base_model_or_path is None:
                prompt_config.base_mode_name_or_path = (
                    self.base_model.__dict__.get('name_or_path', None)
                )

            inference_mode = prompt_config.inference_mode
            prompt_config.inference_mode = True

            auto_mapping_dict = None
            prompt_config.save_pretrained(
                output_dir, auto_mapping_dict=auto_mapping_dict)
            prompt_config.inference_mode = inference_mode

    # DONE
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
            prefix_attention_mask = torch.ones(
                batch_size, prompt_config.num_virtual_tokens).to(decoder_attention_mask.device)

        kwargs.update({
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'position_ids': None,
            'token_type_ids': None
        })

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(
                attention_mask.device
            )
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)

        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat(
            (prompts[:, :, prompt_config.num_virtual_tokens], inputs_embeds), dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs
        )

    # DONE
    def generate(self, **kwargs):
        prompt_config = self._peft_config[self.adapter_name]
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model_prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            kwargs['position_ids'] = None
            kwargs['token_type_ids'] = None

            kwargs = deepcopy(kwargs)

            if 'encoder_outputs' in kwargs:
                del kwargs['encoder_outputs']

            input_ids = kwargs.pop('input_ids', None)
            inputs_embeds = self.word_embeddings(input_ids)
            batch_size = inputs_embeds.shape[0]
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)

            inputs_embeds = torch.cat(
                (prompts[:, :, prompt_config.num_virtual_tokens], inputs_embeds), dim=1)
            kwargs['inouts_embeds'] = inputs_embeds

            if 'attention_mask' in kwargs:
                prefix_attention_mask = torch.ones(batch_size, prompt_config.num_virtual_tokens).to(
                    kwargs['attention_mask'].device
                )
                kwargs['attention_mask'] = torch.cat(
                    (prefix_attention_mask, kwargs['attention_mask']), dim=1)

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

    # DONE
    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name='default', is_trainable=False, config=None, **kwargs):
        config = PromptTuningConfig.from_pretrained(model_id, **kwargs)

        model = cls(model, config, adapter_name=adapter_name)
        model.load_adapter(model_id, adapter_name=adapter_name,
                           is_trainable=is_trainable, **kwargs)
        return model

    # DONE
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
            deepspeed_distributed_tensor_shape = getattr(
                value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                self.word_embeddings = transformer_backbone.get_submodule(
                    named_param.replace(".weight", ""))
                break

        prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        prompt_encoder.to(self.device)

        self.prompt_encoder.update(
            torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    # DONE
    def get_prompt_embedding_to_save(self, adapter_name):
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name]
            .unsqueeze(0)
            .expand(1, -1)
            .to(prompt_encoder.embeddings.weight.device)
        )

        prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    # DONE
    def get_prompt(self, batch_size):
        prompt_config = self._peft_config[self.adapter_name]
        prompt_encoder = self.prompt_encoder[self.adapter_name]

        prompt_tokens = (
            self.prompt_tokens[self.adapter_name]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embeddings.weight.device)
        )
        if prompt_config.inference_mode:
            return prompt_encoder.embeddings.weight.repeat(batch_size, 1, 1)
        else:
            return prompt_encoder(prompt_tokens)

    # DONE
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

    # DONE
    def get_base_model(self):
        return self.base_model

    # DONE
    def load_adapter(self, model_id, adapter_name='default', is_trainable=False, **kwargs):
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()

        adapter_weights = load_adapter_weights(
            model_id, device=torch_device, **hf_hub_download_kwargs)
        load_result = set_peft_model_state_dict(
            self, adapter_weights, adapter_name=adapter_name)
        if (
            (getattr(self, 'hf_device_map', None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({'cpu', 'disk'})) > 0)
            and len(self._peft_config) == 1
        ):
            device_map = kwargs.get('device_map', 'auto')
            max_memory = kwargs.get('max_memory', None)
            offload_dir = kwargs.get('offload_dir', None)
            offload_index = kwargs.get('offload_index', None)

            dispatch_model_kwargs = {}
            no_split_module_classes = self._no_split_modules

            if 'offload_index' in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs['offload_index'] = offload_index

            if device_map != 'sequential':
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == 'balanced_low_0')
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs
            )
            hook = AlignDevicesHook(io_same_device=True)
            remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        if not is_trainable:
            self.eval()
        return load_result

    # DONE
    def add_adapter(self, adapter_name, prompt_config):
        self._peft_config[adapter_name] = prompt_config
        if hasattr(self.config, "to_dict"):
            dict_config = self.config.to_dict()
        else:
            dict_config = self.config

        prompt_config = _prepare_prompt_learning_config(
            prompt_config, dict_config)
        self._setup_prompt_encoder(adapter_name)

    # DONE
    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(
            *args, **kwargs)

        return model_kwargs
