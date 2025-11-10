import os
import json
import torch
import random
import logging
import argparse
import numpy as np

from invoker import Invoker
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_model_tokenizer(args: argparse.Namespace, logger: logging.Logger):
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    # build invoker model
    encoder = model_class.from_pretrained(args.model_name_or_path)
    # add special tokens
    new_special_tokens = ["<last_edit>", "</last_edit>",
                          "<before>", "</before>",
                          "<after>", "</after>",
                          "<previous_edit>", 
                          "</previous_edit>",
                          "<variable_rename>", "<function_rename>", 
                          "<def&ref>", "<clone>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    encoder.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    invoker = Invoker(encoder, config)
    logger.info(f"Set up model type: {args.model_type}")
    
    if args.load_model_path:
        logger.info(f"Load checkpoint from {args.load_model_path}")
        invoker.load_state_dict(torch.load(args.load_model_path))

    return invoker, tokenizer

def get_dataloader_size(args: argparse.Namespace) -> int:
    with open(args.train_filename, "r") as f:
        raw_dataset = json.load(f)

    if args.debug_mode:
        sample_number = min(args.debug_size, len(raw_dataset))
    else:
        sample_number = len(raw_dataset)
    
    return sample_number // args.batch_size

def load_and_tokenize_data(train_filename, tokenizer, args, logger):
    with open(train_filename, "r") as f:
        dataset = json.load(f)
    
    source_ids = []
    source_masks = []
    labels = []
    for sample in dataset:
        if sample["class"] == "variable_rename":
            label_idx = 0
        elif sample["class"] == "function_rename":
            label_idx = 1
        elif sample["class"] == "def&ref":
            label_idx = 2
        elif sample["class"] == "clone":
            label_idx = 3
        else:
            raise ValueError(f"Unknown class type: {sample['input']}")
        
        label = [0,0,0,0]
        label[label_idx] = sample["binary_label"]
        input = tokenizer(sample["input"], padding="max_length", truncation=True, max_length=args.max_source_length)
        label = torch.tensor(label, dtype=torch.float32)
        source_ids.append(torch.tensor(input["input_ids"]))
        source_masks.append(torch.tensor(input["attention_mask"]))
        labels.append(label)
    
    source_ids = torch.stack(source_ids, dim=0)
    source_masks = torch.stack(source_masks, dim=0)
    labels = torch.stack(labels, dim=0)
    
    if args.debug_mode:
        sample_number = min(args.debug_size, len(dataset))
    else:
        sample_number = len(dataset)

    source_ids = source_ids[:sample_number]
    source_masks = source_masks[:sample_number]
    labels = labels[:sample_number]
    dataset = torch.utils.data.TensorDataset(source_ids, source_masks, labels)
        
    logger.info(f"Load {len(dataset)} samples from {train_filename}")
    
    return dataset
