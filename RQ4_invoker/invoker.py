import os
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class Invoker(nn.Module):
    """
        Parameters:

        * `encoder`- encoder. e.g. roberta
        * `config`- configuration of encoder model
    """
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, 4, bias=True)
        
        self.criterion = nn.BCEWithLogitsLoss()
                                   
    def forward(self, source_ids=None, source_mask=None, labels=None, train=True):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        hidden_states = torch.tanh(self.dense(encoder_output)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states).contiguous()
        cls_logits = lm_logits[:,0,:]
        if train:
            loss = self.criterion(cls_logits, labels)
            return loss
        else:
            return cls_logits
        
def train_invoker(args: argparse.Namespace, epoch: int, invoker: Invoker, train_dataloader: DataLoader,
                  eval_dataloader: DataLoader, recording_variables: list, 
                  optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, 
                  logger: logging.Logger) -> list:
    nb_tr_examples, nb_tr_steps, global_step,best_f1 = recording_variables
    tr_loss = 0
    invoker.train()
    
    """
    Training of 1 epoch
    """
    bar = tqdm(train_dataloader,total=len(train_dataloader))
    for batch in bar:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids,source_mask,labels = batch
        loss = invoker(source_ids=source_ids,source_mask=source_mask,labels=labels)

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
    model_to_save = invoker.module if hasattr(invoker, 'module') else invoker  # Only save the model it-self
    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    """
    Eval of 1 epoch
    """
    if args.do_eval and epoch != args.num_train_epochs - 1:
        # only do eval at not last epochs
        logger.info("***** Evaluate locator *****")

        macro_f1, probabilities, predictions, golds = eval_invoker(invoker,eval_dataloader,args)
        # Save best checkpoint for best ppl                    
        if macro_f1 > best_f1:
            logger.info(f"  Best F1:{macro_f1:.4f}")
            logger.info("  "+"*"*20)
            best_f1 = macro_f1
            
            output_dir = os.path.join(args.output_dir, 'checkpoint-best')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = invoker.module if hasattr(invoker, 'module') else invoker  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)  

        invoker.train()

    return [nb_tr_examples, nb_tr_steps, global_step, best_f1]

def eval_invoker(invoker: Invoker, dataloader: DataLoader, args: argparse.Namespace):
    probabilities = []
    predictions = []
    golds = []
    bar = tqdm(dataloader,total=len(dataloader))
    invoker.eval()
    for batch in bar:
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask,labels = batch
            logits = invoker(source_ids=source_ids,source_mask=source_mask,labels=labels, train=False)
            probability = torch.sigmoid(logits)
            binary_predictions = (probability >= 0.5).int()
            probabilities.extend(probability.cpu().numpy())
            predictions.extend(binary_predictions.cpu().numpy())
            golds.extend(labels.cpu().numpy())
    
    invoker.train()
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    golds = np.array(golds)
    
    macro_f1 = f1_score(golds, predictions, average='macro')
    
    return macro_f1, probabilities, predictions, golds