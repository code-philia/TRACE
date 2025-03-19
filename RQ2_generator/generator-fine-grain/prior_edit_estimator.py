import os
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Union
from dependency_analyzer import load_dep_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

class Estimator(nn.Module):
    def __init__(self, embed_model_name: str, dependency_model: RobertaModel) -> None:
        super().__init__()
        
        self.config = RobertaConfig.from_pretrained(embed_model_name)
        self.hidden_size = self.config.hidden_size
        self.dependency_model = dependency_model
        self.sigmoid = nn.Sigmoid()
        # freeze dependency model
        for param in self.dependency_model.parameters():
            param.requires_grad = False
        self.embed_model = RobertaModel.from_pretrained(embed_model_name)
        self.compress = nn.Sequential(
            nn.Linear(self.hidden_size*2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        self.output = nn.Sequential(
            nn.Linear(18, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
            
    def forward(self, code_distance: torch.tensor, 
                sep_input_ids: torch.tensor, sep_attn_masks: torch.tensor,
                combine_input_ids: torch.tensor, combine_attn_masks: torch.tensor) -> torch.Tensor:
        '''
        Args:
            code_distance: (batch_size, 1)
            sep_input_ids: (batch_size, 2, max_length), dim 1: 0 for target code, 1 for edit
            sep_attn_masks: (batch_size, 2, max_length), dim 1: 0 for target code, 1 for edit
            combine_input_ids: (batch_size, max_length), combined input_ids of target code and edit
            combine_attn_masks: (batch_size, max_length), combined attn_masks of target code and edit
        '''
        # embeddings: (batch_size(1), 2, hidden_size)
        # embed_model only take shape (batch_size, max_length) tensor, so we need to stack them by dim 0
        embeddings = torch.stack([self.embed_model(sep_input_ids[i], attention_mask=sep_attn_masks[i]).last_hidden_state[:, 0, :] for i in range(sep_input_ids.shape[0])], dim=0) 
        
        # get dependency score
        dep_output = self.dependency_model(combine_input_ids, combine_attn_masks)[:,-1:]
        dep_score = self.sigmoid(dep_output)
        
        # embeddings: (batch_size, 2*hidden_size)
        embeddings = torch.cat([embeddings[:,0,:], embeddings[:,1,:]], dim=1)
        
        # similarity: (batch_size, 16)
        similarity = self.compress(embeddings) 

        # output: (batch_size, 1)
        output = self.output(torch.cat([dep_score, code_distance, similarity], dim=-1))
        return output

def load_estimator(args: argparse.Namespace) -> tuple[Estimator, RobertaTokenizer]:
    dependency_model, dependency_tokenizer = load_dep_model(args)
    estimator = Estimator("huggingface/CodeBERTa-small-v1", dependency_model) 
    estimator_tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    if args.load_estimator_model_path is not None:
        estimator.load_state_dict(torch.load(args.load_estimator_model_path))
    return estimator, estimator_tokenizer, dependency_tokenizer
   
def load_estimator_data(dataset: list[dict], estimator_tokenizer: RobertaTokenizer, 
        dependency_tokenizer: RobertaTokenizer, args: argparse.Namespace) -> TensorDataset:
    """
    Func:
        give a list of data sample, convert them to TensorDataset for estimator
    Args:
        dataset: list[dict], dict of key: "sliding_window", "prior_edit", "code_distance", 
            "html_url" (optional), "commit_msg" (optional), "label" (optional);
        estimator_tokenizer: RobertaTokenizer, tokenizer for estimator
        dependency_tokenizer: RobertaTokenizer, tokenizer for dependency analyzer
    Return:
        dataset: TensorDataset, dataset for estimator
    """
    distance_scores = []
    sep_inputs = []
    sep_attn_masks = []
    combine_inputs = []
    combine_attn_masks = []
    for sample in dataset:
        distance_scores.append(torch.tensor([sample["code_distance"]], dtype=torch.float32))
        
        target_code = "".join(sample["sliding_window"].before_edit_window())
        prior_edit = sample["prior_edit"].before_edit_window(split_by_line=False) + sample["prior_edit"].after_edit_window(split_by_line=False)
        sep_input = [target_code, prior_edit]
        sep_input = estimator_tokenizer(sep_input, padding="max_length", truncation=True, max_length=args.max_source_length, return_tensors="pt")
        sep_inputs.append(sep_input["input_ids"])
        sep_attn_masks.append(sep_input["attention_mask"])
        
        combine_input = "<from>" + target_code + "<to>" + prior_edit
        combine_input = dependency_tokenizer(combine_input, padding="max_length", truncation=True, max_length=args.max_source_length, return_tensors="pt")
        combine_inputs.append(combine_input["input_ids"])
        combine_attn_masks.append(combine_input["attention_mask"])
    
    distance_scores = torch.stack(distance_scores, dim = 0)
    sep_inputs = torch.stack(sep_inputs, dim=0)
    sep_attn_masks = torch.stack(sep_attn_masks, dim=0)
    combine_inputs = torch.cat(combine_inputs, dim=0)
    combine_attn_masks = torch.cat(combine_attn_masks, dim=0)
    
    dataset = TensorDataset(distance_scores, sep_inputs, sep_attn_masks, combine_inputs, combine_attn_masks)
    
    return dataset

def train_estimator(model: Estimator, train_dataloader: DataLoader, dev_dataloader: DataLoader,
                    device: torch.device, epoch: int, args: argparse.Namespace, recording_variables_estimator: list,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, logger: logging.Logger) -> None:
    """
    Func:
        Train 1 epoch of estimator.
    """
    best_loss = recording_variables_estimator[0]
    model.train()
    
    """
    Training of 1 epoch
    """
    logger.info("***** Train Estimator *****")
    pbar = tqdm(train_dataloader, desc=f"Train epoch: {epoch}")
    for batch in pbar:
        distance_scores, sep_inputs, sep_attn_masks, combine_inputs, combine_attn_masks, gold = [b.to(device) for b in batch]
        output = model(distance_scores, sep_inputs, sep_attn_masks, combine_inputs, combine_attn_masks)
        loss = criterion(output, gold.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
    
    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)
    torch.save(model.state_dict(), os.path.join(last_output_dir, f"estimator.bin"))
    
    """
    eval of 1 epoch
    """
    # evaluate
    logger.info("***** Evaluate Estimator *****")
    loss, preds, golds = evaluate_estimator(model, dev_dataloader, "validation", logger)

    # save estimator's validation results
    with open(os.path.join(args.output_dir, f"estimator_val_{epoch}.txt"), "w") as f:
        for pred, gold in zip(preds, golds):
            f.write(f"pred: {pred:.4f}\tgold: {gold:.4f}\n")

    # save model
    if loss < best_loss:
        logger.info(f"  Best loss: {round(loss,5)}")
        logger.info("  "+"*"*20)
        best_output_dir = os.path.join(args.output_dir, 'checkpoint-best')
        if not os.path.exists(best_output_dir):
            os.makedirs(best_output_dir)
        best_loss = loss
        torch.save(model.state_dict(), os.path.join(best_output_dir, f"estimator.bin"))

    return [best_loss]
    
def evaluate_estimator(model: Estimator, dataloader: DataLoader, mode: str, logger: Union[logging.Logger, None]=None, show_progress: bool = True) -> Union[np.array, tuple[float, np.array, np.array]]:
    """
    Args:
        model: Estimator
        dataloader: DataLoader
        mode: str, "validation", "test" or "infer", in mode "infer", the dataloader should not contain golds
    Return:
        preds: np.array, (num_samples, )
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    golds = []
    preds = []
    if show_progress:
        dataloader = tqdm(dataloader, desc="Evaluating")
    for batch in dataloader:
        if mode != 'infer':
            distance_scores, sep_inputs, sep_attn_masks, combine_inputs, combine_attn_masks, gold = [b.to(device) for b in batch]
            golds.extend(gold.detach().cpu())
        else:
            distance_scores, sep_inputs, sep_attn_masks, combine_inputs, combine_attn_masks = [b.to(device) for b in batch]
        
        output = model(distance_scores, sep_inputs, sep_attn_masks, combine_inputs, combine_attn_masks)
        preds.extend(output.detach().cpu())

    if mode != "infer":
        golds = torch.tensor(golds, dtype=torch.float32).numpy()
        preds = torch.tensor(preds, dtype=torch.float32).numpy()

        # MSELoss
        mseloss = np.mean((golds - preds)**2)
        logger.info(f"MSE loss = {mseloss}")
        return mseloss, preds, golds
    else:
        preds = torch.tensor(preds, dtype=torch.float32).numpy()
        
    return preds