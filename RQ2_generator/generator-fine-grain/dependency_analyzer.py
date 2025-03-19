import os
import torch

import numpy as np
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import EncoderDecoderModel, RobertaTokenizerFast, PreTrainedModel
from torch.utils.data import DataLoader, TensorDataset

class DependencyAnalyzer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, encoder: PreTrainedModel | None = None,
                 match_tokenizer: RobertaTokenizerFast | None = None):
        super(DependencyAnalyzer, self).__init__()
        if not encoder:
            encoder: PreTrainedModel = EncoderDecoderModel.from_encoder_decoder_pretrained("microsoft/codebert-base", "microsoft/codebert-base").encoder
        if match_tokenizer:
            encoder.resize_token_embeddings(len(match_tokenizer))
            encoder.config.decoder_start_token_id = match_tokenizer.cls_token_id
            encoder.config.pad_token_id = match_tokenizer.pad_token_id
            encoder.config.eos_token_id = match_tokenizer.sep_token_id
            encoder.config.vocab_size = match_tokenizer.vocab_size
        self.encoder = encoder
        self.dense = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        output_2d = self.dense(pooler_output)
        return output_2d
    
def load_dep_model(args):
    tokenizer = RobertaTokenizerFast.from_pretrained(args.load_dep_model_path)
    model = DependencyAnalyzer(match_tokenizer=tokenizer)
    model.load_state_dict(torch.load(os.path.join(args.load_dep_model_path,'pytorch_model.bin')))
    return model, tokenizer
