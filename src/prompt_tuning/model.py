import torch
from transformers import AutoTokenizer
import math
from .config import PromptTuningInit


class PromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()
        init_type = config.init_type

        total_virtual_tokens = config.num_virtual_tokens * \
            config.num_transformer_submodules
        self.embeddings = torch.nn.Embedding(
            total_virtual_tokens, config.token_dim)
        if init_type == PromptTuningInit.TEXT:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            init_text = config.init_text
            init_tokens_ids = tokenizer(init_text)['input_ids']
            num_tokens = len(init_tokens_ids)
            if num_tokens > total_virtual_tokens:
                init_tokens_ids = init_tokens_ids[:total_virtual_tokens]
            elif num_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_tokens)
                init_tokens_ids = init_tokens_ids * num_reps

            init_tokens_ids = init_tokens_ids[:total_virtual_tokens]
            init_tokens_ids = torch.LongTensor(
                init_tokens_ids).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                init_tokens_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            word_embedding_weights = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        return self.embeddings(indices)
