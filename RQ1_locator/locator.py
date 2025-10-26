# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class Locator(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder. e.g. roberta
        * `config`- configuration of encoder model. 
        * `mask_id`- the id of mask token. e.g. 50264
    """
    def __init__(self, encoder, config, 
                 inline_mask_id=None, inter_mask_id=None, 
                 keep_token_id=None, delete_token_id=None, replace_token_id=None, 
                 null_token_id=None, insert_token_id=None, block_split_token_id=None,
                 args=None):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.model_type = args.model_type
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.inline_mask_id=inline_mask_id
        self.inter_mask_id=inter_mask_id
        self.keep_token_id=keep_token_id
        self.delete_token_id=delete_token_id
        self.replace_token_id=replace_token_id
        self.null_token_id=null_token_id
        self.insert_token_id=insert_token_id
        self.block_split_token_id=block_split_token_id
        self.label_weight = torch.ones(config.vocab_size) * 1e-3
        self.label_weight[keep_token_id] = args.keep_weight
        self.label_weight[delete_token_id] = args.insert_weight 
        self.label_weight[replace_token_id] = args.replace_weight
        self.label_weight[null_token_id] = args.null_weight
        self.label_weight[insert_token_id] = args.insert_weight
        self.label_weight[block_split_token_id] = args.block_split_weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=self.label_weight)
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        if self.model_type == "codet5":
            # T5 encoder has different embedding module
            self._tie_or_clone_weights(self.lm_head,
                                    self.encoder.embed_tokens)
        else:
            self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)  
                                   
    def forward(self, source_ids=None, source_mask=None, target_ids=None, train=True):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        hidden_states = torch.tanh(self.dense(encoder_output)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states).contiguous()
        if train:
            # Flatten the tokens
            active_loss = ((source_ids == self.inter_mask_id) | (source_ids == self.inline_mask_id)).contiguous().view(-1) # find which tokens are masked
            labels = target_ids.contiguous().view(-1)[active_loss] # get the labels of the masked tokens
            filtered_logits = lm_logits.contiguous().view(-1, self.config.vocab_size)[active_loss] # get the logits of the masked tokens

            loss = self.criterion(filtered_logits, labels)
            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            return lm_logits
        
# compute the ppl loss and 4 metric
def compute_eval_loss_metric(model: Locator, tokenizer, eval_dataloader: DataLoader, device: torch.device) -> tuple[float, np.array, str]:
    model.eval()

    eval_loss,tokens_num = 0,0
    pred_labels = []
    gold_labels = []

    for batch in tqdm(eval_dataloader, desc="Locator evaluation"):
        source_ids,source_mask,target_ids = [b.to(device) for b in batch]                  

        with torch.no_grad():
            lm_logits = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,train=False)
            active_loss = ((source_ids == model.inter_mask_id) | (source_ids == model.inline_mask_id)).contiguous().view(-1) # find which tokens are masked
            labels = target_ids.contiguous().view(-1)[active_loss] # get the labels of the masked tokens
            filtered_logits = lm_logits.contiguous().view(-1, model.config.vocab_size)[active_loss] # get the logits of the masked tokens

            loss_fct = model.criterion
            loss = loss_fct(filtered_logits, labels)
            loss = loss * active_loss.sum()
            num = active_loss.sum()

        # compute loss
        eval_loss += loss.sum().item()
        tokens_num += num.sum().item()

        pred_labels.append(torch.argmax(filtered_logits,dim=-1))
        gold_labels.append(labels)

    pred_label_ids = torch.cat(pred_labels,dim=0).cpu()
    gold_label_ids = torch.cat(gold_labels,dim=0).cpu()

    # convert to actual labels
    pred_labels = tokenizer.convert_ids_to_tokens(pred_label_ids)
    gold_labels = tokenizer.convert_ids_to_tokens(gold_label_ids)
    
    acc = accuracy_score(gold_labels, pred_labels)
    precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(gold_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(gold_labels, pred_labels, average='macro', zero_division=0)
    report = classification_report(gold_labels, pred_labels, digits=4, zero_division=0)

    pred_result = np.array([acc, precision, recall, f1])
    eval_ppl = eval_loss / tokens_num
    return eval_ppl, pred_result, report

def train_locator(args: argparse.Namespace, epoch: int, locator: Locator, tokenizer, train_dataloader: DataLoader,
                  eval_dataloader: DataLoader, device: torch.device, recording_variables: list, 
                  optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, 
                  logger: logging.Logger) -> list:
    """
    Func:
        Train 1 epoch of locator.
    """
    nb_tr_examples, nb_tr_steps,global_step,best_ppl = recording_variables
    tr_loss = 0
    locator.train()
    
    """
    Training of 1 epoch
    """
    bar = tqdm(train_dataloader,total=len(train_dataloader))
    for batch in bar:
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask,target_ids = batch
        loss,_,_ = locator(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids)

        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        tr_loss += loss.item()
        train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
        bar.set_description("epoch {} loss {}".format(epoch,train_loss))
        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
            #Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

    # save last checkpoint
    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)
    model_to_save = locator.module if hasattr(locator, 'module') else locator  # Only save the model it-self
    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # save checkpoint at each epoch
    output_dir = os.path.join(args.output_dir, 'checkpoint-each-epoch')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = locator.module if hasattr(locator, 'module') else locator  # Only save the model it-self
    output_model_file = os.path.join(output_dir, f"pytorch_model_{epoch}.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    """
    Eval of 1 epoch
    """
    if args.do_eval:
        logger.info("***** Evaluate locator *****")
        logger.info("  Batch size = %d", args.locator_batch_size)

        eval_ppl, pred_result, report = compute_eval_loss_metric(locator,tokenizer,eval_dataloader,device)

        # print eval results
        result = {  'eval_ppl': round(np.exp(eval_ppl),5),
                    'Accuracy': round(np.mean(pred_result[0]),4),
                    'Precision': round(np.mean(pred_result[1]),4),
                    'Recall': round(np.mean(pred_result[2]),4),
                    'F1': round(np.mean(pred_result[3]),4)
                    }
        
        logger.info(f"Classification report:\n{report}")
        for key in result.keys():
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  "+"*"*20)
        
        # Save best checkpoint for best ppl                    
        if eval_ppl < best_ppl:
            logger.info("  Best ppl:%s",round(np.exp(eval_ppl),5))
            logger.info("  "+"*"*20)
            best_ppl = eval_ppl
            
            output_dir = os.path.join(args.output_dir, 'checkpoint-best')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = locator.module if hasattr(locator, 'module') else locator  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)  

        locator.train()

    return [nb_tr_examples, nb_tr_steps, global_step, best_ppl]

def evaluate_locator(model: Locator, dataloader: DataLoader, device: torch.device, 
                     locator_tokenizer: RobertaTokenizer, logger: logging.Logger, args: argparse.Namespace) -> tuple[list, list]:
    """
    Func:
        Given a model and a dataloader, evaluate the model on the dataloader.
    Args:
        model: the locator model to be evaluated.
        dataloader: the dataloader to be evaluated on.
        device: the device to run the model on.
        locator_tokenizer: the tokenizer used to decode the data.
        logger: the logger to log the evaluation result.
    return:
        pred_edit_labels: list[list], len(pred_edit_labels) = num_examples, 
            each element is a list of edit operations: <keep>, <add>, <replace>
        gold_edit_labels: list[list], len(gold_edit_labels) = num_examples,
            each element is a list of edit operations: <keep>, <add>, <replace>
    """
    model.to(device)
    model.eval()
    
    pred_edit_labels = []
    gold_edit_labels = []
    confidences = []
    softmax = torch.nn.Softmax(dim=-1)

    for batch in tqdm(dataloader,total=len(dataloader), desc="Locator evaluation"):
        source_ids,source_mask,target_ids = [b.to(device) for b in batch]
        with torch.no_grad():
            lm_logits = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,train=False).to('cpu')
            # extract masked edit operations
            for i in range(lm_logits.shape[0]): # for sample within batch
                output = []
                gt = []
                confidence = []
                for j in range(lm_logits.shape[1]): # for every token
                    if source_ids[i][j] == model.inline_mask_id or source_ids[i][j] == model.inter_mask_id: # if is masked
                        # save the confidence of 6 labels:
                        # <keep>, <add>, <replace>, <null>, <insert>, <block_split>
                        softmax_output = softmax(lm_logits[i][j])
                        token_confidence = [softmax_output[model.keep_token_id].item(),
                                            softmax_output[model.delete_token_id].item(),
                                            softmax_output[model.replace_token_id].item(),
                                            softmax_output[model.null_token_id].item(),
                                            softmax_output[model.insert_token_id].item(),
                                            softmax_output[model.block_split_token_id].item()]
                        confidence.append(token_confidence)
                        output.append(locator_tokenizer.decode(torch.argmax(lm_logits[i][j]),clean_up_tokenization_spaces=False))
                        gt.append(locator_tokenizer.decode(target_ids[i][j],clean_up_tokenization_spaces=False))
                pred_edit_labels.append(output.copy())
                gold_edit_labels.append(gt.copy())
                confidences.append(confidence)
        
    # label process
    pred_edit_labels, confidences = hard_rule_label_correction(pred_edit_labels, confidences, args.label_num)
    all_preds = []
    all_gts = []
    for i in range(len(pred_edit_labels)):
        all_preds.extend(pred_edit_labels[i])
        all_gts.extend(gold_edit_labels[i])

    return pred_edit_labels, gold_edit_labels, confidences

def locator_loss_by_sample(model: Locator, dataloader: DataLoader, device: torch.device, 
                     locator_tokenizer: RobertaTokenizer) -> list[torch.tensor]:
    """
    Func:
        Given a model and a dataloader, evaluate the model on the dataloader.
    Args:
        model: the locator model to be evaluated.
        dataloader: the dataloader to be evaluated on.
        device: the device to run the model on.
        locator_tokenizer: the tokenizer used to decode the data.
    return:
        losses: list of length len(dataloader), each element is a tensor of shape (#masks)
    """
    model.to(device)
    model.eval()

    criterion = model.criterion
    
    losses = []
    for batch in tqdm(dataloader,total=len(dataloader), desc="Locator evaluation"):
        source_ids,source_mask,target_ids = [b.to(device) for b in batch]
        with torch.no_grad():
            lm_logits = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,train=False)
            # extract masked edit operations
            for i in range(lm_logits.shape[0]): # for sample within batch
                output = []
                gt = []
                for j in range(lm_logits.shape[1]): # for every token
                    if source_ids[i][j] == model.inline_mask_id or source_ids[i][j] == model.inter_mask_id: # if is masked
                        output.append(lm_logits[i][j])
                        gt.append(target_ids[i][j])
                output = torch.stack(output) # (#masks, vocab_size)
                gt = torch.stack(gt) # ( #masks)
                loss = criterion(output,gt).detach()
                losses.append(torch.tanh(loss))
    return losses

def hard_rule_label_correction(all_preds: list[list[str]], all_confidences: list[list[list[float]]], label_num: int):
    """
    Func:
        Correct the labels based on the hard rules.
    Args:
        all_preds: list[list[str]], len(all_preds) = num_examples, len(all_preds[i]) = num_masks in example i, each element is a label
        all_confidences: list[list[list[float]]], len(confidences) = num_examples, len(confidences[i]) = num_masks in example i, len(confidences[i][j]) = 6, represents the confidence of 6 labels
    return:
        all_preds: list[list[str]], len(all_preds) = num_examples, len(all_preds[i]) = num_masks in example i, each element is a label
        all_confidences: list[list[float]], len(confidences) = num_examples, len(confidences[i]) = num_masks in example i, confidence[i][j] = confidence of the predicted label
    """
    if label_num == 6:
        for sample_idx, preds in enumerate(all_preds):
            if preds[0] == "<block-split>":
                preds[0] = "<null>"
            if preds[-1] == "<block-split>":
                preds[-1] = "<null>"
            for label_idx, label in enumerate(preds[1:-1], start=1):
                # <block-split> should be surrounded by <replace>
                if label == "<block-split>" and (preds[label_idx-1] != "<replace>" or preds[label_idx+1] != "<replace>"):
                    preds[label_idx] = "<null>"
                
            # if there are multiple <insert>, you can't have all <delete> within them
            # get the index of <insert>
            insert_idxs = [i for i, label in enumerate(preds) if label == "<insert>"]
            if len(insert_idxs) <= 1:
                continue
            for i in range(len(insert_idxs)-1):
                insert_begin_idx = insert_idxs[i]
                insert_end_idx = insert_idxs[i+1]
                all_delete = True
                for label in preds[insert_begin_idx+1:insert_end_idx]:
                    if label == "<keep>" or label == "<replace>":
                        all_delete = False
                        break
                
                if all_delete: # we need to change one <insert> to <null>
                    start_insert_confidence = all_confidences[sample_idx][insert_begin_idx][4]
                    end_insert_confidence = all_confidences[sample_idx][insert_end_idx][4]
                    if start_insert_confidence > end_insert_confidence:
                        preds[insert_end_idx] = "<null>"
                    else:
                        preds[insert_begin_idx] = "<null>"
            
    for preds, confidences in zip(all_preds, all_confidences):
        for i in range(len(preds)):
            if preds[i] == "<keep>":
                confidences[i] = confidences[i][0]
            elif preds[i] == "<delete>":
                confidences[i] = confidences[i][1]
            elif preds[i] == "<replace>":
                confidences[i] = confidences[i][2]
            elif preds[i] == "<null>":
                confidences[i] = confidences[i][3]
            elif preds[i] == "<insert>":
                confidences[i] = confidences[i][4]
            elif preds[i] == "<block-split>":
                confidences[i] = confidences[i][5]
            else:
                raise ValueError(f"Unknown label: {preds[i]}")
    
                
    return all_preds, all_confidences
            