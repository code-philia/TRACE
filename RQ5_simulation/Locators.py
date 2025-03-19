import os
import time
import torch
import argparse

import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from rank_bm25 import BM25Okapi
from code_window import CodeWindow
from is_clone import find_clone_in_project
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
from CoEdPilot_estimator import load_estimator_data, evaluate_estimator


class TRACELocator(nn.Module):
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
                 null_token_id=None, insert_token_id=None, block_split_token_id=None):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.model_type = "codet5"
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
        
class CoEdPilotLocator(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,config, beam_size=None,max_length=None,sos_id=None,eos_id=None,mask_id=None):
        super(CoEdPilotLocator, self).__init__()
        self.encoder = encoder
        self.config=config
        self.model_type = "codet5"
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.mask_id=mask_id
        self.label_weight = torch.zeros(config.vocab_size)
        self.label_weight[self.mask_id] = 1.0
        
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
                                   
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,train=True):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        hidden_states = torch.tanh(self.dense(encoder_output)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states).contiguous()
        if train:
            # Flatten the tokens
            active_loss = (source_ids == self.mask_id).contiguous().view(-1) # find which tokens are masked
            labels = target_ids.contiguous().view(-1)[active_loss] # get the labels of the masked tokens
            filtered_logits = lm_logits.contiguous().view(-1, self.config.vocab_size)[active_loss] # get the logits of the masked tokens

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(filtered_logits, labels)
            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            return lm_logits
        
def load_locator(args, logger):
    if args.system in ["TRACE", "TRACE-wo-Invoker", "EnrichedSemantics", "PlainSemantics"]:
        """
        TRACE, TRACE-wo-Invoker, EnrichedSemantics are using the same locator model
        PlainSemantics is using a different model, but share same loading method
        """
        config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
        locator_config = config_class.from_pretrained('salesforce/codet5-large')
        locator_tokenizer = tokenizer_class.from_pretrained('salesforce/codet5-large')
        encoder = model_class.from_pretrained('salesforce/codet5-large').encoder

        # add special tokens
        new_special_tokens = ["<inter-mask>",
                            "<code_window>", "</code_window>", 
                            "<prompt>", "</prompt>", 
                            "<prior_edits>", "</prior_edits>",
                            "<edit>", "</edit>",
                            "<keep>", "<replace>", "<delete>",
                            "<null>", "<insert>", "<block-split>",
                            "</insert>","<replace-by>", "</replace-by>"]
        locator_tokenizer.add_tokens(new_special_tokens, special_tokens=True)
        encoder.resize_token_embeddings(len(locator_tokenizer))
        locator_config.vocab_size = len(locator_tokenizer)
        
        locator=TRACELocator(encoder=encoder,config=locator_config,
                        inline_mask_id=locator_tokenizer.mask_token_id,
                        inter_mask_id=locator_tokenizer.convert_tokens_to_ids("<inter-mask>"),
                        keep_token_id=locator_tokenizer.convert_tokens_to_ids("<keep>"),
                        delete_token_id=locator_tokenizer.convert_tokens_to_ids("<delete>"),
                        replace_token_id=locator_tokenizer.convert_tokens_to_ids("<replace>"),
                        null_token_id=locator_tokenizer.convert_tokens_to_ids("<null>"),
                        insert_token_id=locator_tokenizer.convert_tokens_to_ids("<insert>"),
                        block_split_token_id=locator_tokenizer.convert_tokens_to_ids("<block-split>"))
        locator.load_state_dict(torch.load(args.locator_model_path, map_location = args.device), strict = False)
        locator.eval()
        locator.to(args.device)
        logger.info(f"Successfully loaded Locator model from: {args.locator_model_path}")
        return locator,locator_tokenizer
    
    elif args.system == "CoEdPilot":
        config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
        locator_config = config_class.from_pretrained('salesforce/codet5-large')
        locator_tokenizer = tokenizer_class.from_pretrained('salesforce/codet5-large')
        encoder = model_class.from_pretrained('salesforce/codet5-large').encoder

        locator_config.vocab_size = len(locator_tokenizer)
        locator=CoEdPilotLocator(encoder,config=locator_config,
                    max_length=512,
                    sos_id=locator_tokenizer.cls_token_id,eos_id=locator_tokenizer.sep_token_id,mask_id=locator_tokenizer.mask_token_id)
        locator.load_state_dict(torch.load(args.locator_model_path, map_location = args.device), strict = False)
        locator.to(args.device)
        logger.info(f"Successfully loaded line locator model from: {args.locator_model_path}")
        return locator,locator_tokenizer
    
    else: # CodeCloneDetector
        return None, None
 
def detect_next_clone(commit, record):
    predictions = {}
    commit.get_current_version(save=True)
    
    start = time.time()
    last_edit = commit.enriched_prev_edits[-1]
    query = last_edit.before_edit_region(split_by_line=False, allow_fuzzy=False)
    clone_detected_locations = find_clone_in_project(commit, query, threshold=80)
    end = time.time()
    record.locator_runtime[-1] += end - start
    
    for file_path in commit.changed_files:
        with open(os.path.join(commit.project_dir, file_path), "r") as f:
            content = f.readlines()
        predictions[file_path] = {
            "inline_predictions": ["<keep>"] * len(content),
            "inline_confidences": [1.0] * len(content)
        }
        for clone_detected_location in clone_detected_locations:
            if clone_detected_location["file_path"] == file_path:
                for matched_line in clone_detected_location["matched_lines"]:
                    predictions[file_path]["inline_predictions"][matched_line] = "<replace>"
    return predictions

def predict_next_location(commit, models, args, record, logger, LSP=None):
    """
    Given the files that will be edited in the commit, 
    predict the next edit operation label for each line of code
    For TRACE, TRACE-wo-Invoker, EnrichedSemantics, PlainSemantics, CoEdPilot
    
    Args:
        commit: the commit object
        models: dict of loaded models
        args: the args object
        record: the record object
        logger: the logger object
        LSP: the LSP object
    """
    # STEP 1: first try TRACE first
    if args.system in ["TRACE", "TRACE-wo-Invoker"]:
        from TRACE import TRACE
        TRACE_predictions = TRACE(commit, models, args, record, logger, LSP=LSP)
        if TRACE_predictions is not None and len(TRACE_predictions) > 0:       
            return TRACE_predictions
    
    # STEP 2: then try sliding window
    commit.get_current_version(save=True)
    predictions = {}
    for file_path in tqdm(commit.changed_files, 
   desc=f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} - INFO - __main__ -   ==> Scanning files for next edit location"):
        abs_file_path = os.path.join(commit.project_dir, file_path)
        with open(abs_file_path, "r") as f:
            content = f.readlines()
        
        # Split content into windows of 10 lines of code
        sliding_windows = split_file_into_windows(content, models["locator_tokenizer"])
        
        # Select prior edits for each window and form inputs
        if args.system != "CoEdPilot":
            # For TRACE, TRACE-wo-Invoker, EnrichedSemantics, PlainSemantics
            dataset = TRACE_make_locator_dataset(sliding_windows, commit, models, args)
        else:
            # For CoEdPilot
            dataset = CoEdPilot_make_locator_dataset(sliding_windows, commit, file_path, models, args)
        
        dataloader = DataLoader(dataset, batch_size=args.locator_batch_size, shuffle=False)
        locator = models["locator"]
        locator_tokenizer = models["locator_tokenizer"]
        locator_results, time_cost = locator_predict(locator, locator_tokenizer, dataloader, args)
        record.locator_runtime[-1] += time_cost
        
        assert len(locator_results["inline_predictions"]) == len(content)
        predictions[file_path] = locator_results
        
    return predictions
    
def TRACE_make_locator_dataset(sliding_windows, commit, models, args):
    """
    This function help each sliding window to find syntactically similar prior edits, then form the input sequence for locator
    For TRACE, TRACE-wo-Invoker, EnrichedSemantics, PlainSemantics
    
    Args:
        sliding_windows: list[list[str]], the sliding windows of code
        commit: the commit object, see `commit.py` for more details
        models: dict of loaded models, should contain key "locator_tokenizer"
        args: the args object
    """
    source_seqs = []
    locator_tokenizer = models["locator_tokenizer"]
    hunks = commit.enriched_prev_edits
    for sliding_window in sliding_windows:
        non_overlap_hunks = hunks
        choosen_hunk_ids = [hunk.idx for hunk in hunks] # index to hunk idx
        tokenized_corpus = [locator_tokenizer.tokenize("".join(hunk.before_edit_region()+hunk.after_edit_region())) for hunk in non_overlap_hunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = locator_tokenizer.tokenize("".join(sliding_window))
        retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3) 
        retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
        prior_edit_idx = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk idx
        prior_edits = []
        for idx in prior_edit_idx: # preserve the order
            prior_edits.append([hunk for hunk in hunks if hunk.idx == idx][0])
            
        source_seq = TRACE_formalize_locator_input(sliding_window, commit.commit_message, prior_edits, locator_tokenizer, args)
        # if args.debug: # Quick check for input format
        #     print(source_seq)
        #     raise Exception("stop")
        source_seqs.append(source_seq)
        
    encoded_source_seq = locator_tokenizer(source_seqs, padding="max_length", truncation=True, max_length=512)
    
    source_ids = torch.tensor(encoded_source_seq["input_ids"])
    source_mask = torch.tensor(encoded_source_seq["attention_mask"])
    dataset = TensorDataset(source_ids, source_mask)

    return dataset

def CoEdPilot_make_locator_dataset(sliding_windows, commit, sw_file_path, models, args):
    """
    Func:
        Given a fixed prior edit estimator, select most relevant hunk as prior edit 
        and construct the dataset for locator to infer
    Args:
        sliding_windows: sliding windows of one file
        commit: the commit object, see `commit.py` for more details
        sw_file_path: the file path of the sliding window
        models: dict of loaded models, should contain key "locator_tokenizer"
        args: the args object
    """    
    estimator = models["estimator"]
    estimator_tokenizer = models["estimator_tokenizer"]
    dependency_tokenizer = models["dependency_tokenizer"]
    locator_tokenizer = models["locator_tokenizer"]
    
    # for each sliding_window, use estimator to select prior_edits
    source_seqs = []
    for idx, sliding_window in enumerate(sliding_windows):
        # form estimator dataset
        estimator_dataset = []

        # given a sliding window, form (sliding_window, hunk) pair
        choosen_hunk_ids = []
        for hunk in commit.enriched_prev_edits:
            choosen_hunk_ids.append(hunk.idx)
            sample = {}
            sample["sliding_window"] = sliding_window
            sample["prior_edit"] = hunk
            sw_start_line_idx = idx * 10
            hunk_at_file = commit.map[hunk.idx]["at_file"]
            hunk_start_line_idx = hunk.edit_start_line_idx
            if sw_file_path == hunk_at_file:
                sample["code_distance"] = max(0, 1 - abs(sw_start_line_idx - hunk_start_line_idx) / 50)
            else:
                sample["code_distance"] = 0
            estimator_dataset.append(sample)

        if len(estimator_dataset) != 0:
            # convert to estimator dataloader
            estimator_dataset = load_estimator_data(estimator_dataset, estimator_tokenizer, dependency_tokenizer)
            estimator_dataloader = DataLoader(estimator_dataset, batch_size=20, shuffle=False)
            # get the prior edit estimation
            estimation = evaluate_estimator(estimator, estimator_dataloader, "infer")
            # project hunk id to estimation score
            scores = []
            for hunk_id, score in zip(choosen_hunk_ids, estimation):
                scores.append([hunk_id, score])
            # sort in descending order
            scores.sort(key=lambda x: x[1], reverse=True)
            # get top 3 prior edits
            top_3_prior_edits = scores[:3]
            prior_edits = []
            for hunk_id in top_3_prior_edits:
                prior_edits.extend([hunk for hunk in commit.enriched_prev_edits if hunk.idx == hunk_id[0]])
        else:
            prior_edits = []

        # for picked prior edits, combine with sliding window, turn into locator data sample
        source_seq = CoEdPilot_formalize_locator_input(sliding_window, commit.commit_message, prior_edits, locator_tokenizer, args)
        # if args.debug: # Quick check for input format
        #     print(source_seq)
        #     raise Exception("stop")
        source_seqs.append(source_seq)
        
    encoded_source_seq = locator_tokenizer(source_seqs, padding="max_length", truncation=True, max_length=512)
    source_ids = torch.tensor(encoded_source_seq["input_ids"])
    source_mask = torch.tensor(encoded_source_seq["attention_mask"])
    dataset = TensorDataset(source_ids, source_mask)
    
    return dataset

def TRACE_formalize_locator_input(sliding_window: list[str], prompt: str, prior_edits: list[CodeWindow], 
                                    tokenizer: RobertaTokenizer, args: argparse.Namespace) -> str:
    """
    Func:
        Given a sliding window, prior edits, and prompt, form the input sequence for locator
        For TRACE, TRACE-wo-Invoker, EnrichedSemantics, PlainSemantics
    
    Args:
        sliding_window: list[str], one sliding window of code
        prompt: str, the commit message
        prior_edits: list[CodeWindow], the prior edit hunks selected
        tokenizer: RobertaTokenizer, the tokenizer of locator model
        args: argparse.Namespace, the args object
    
    Returns:
        source_seq: str, the input sequence for locator
    """
    if args.label_num == 6:
        # For TRACE, TRACE-wo-Invoker, EnrichedSemantics
        source_seq = "<code_window><inter-mask>"
        for line_of_code in sliding_window:
            source_seq += f"<mask>{line_of_code}<inter-mask>"
    else:
        # For PlainSemantics
        source_seq = "<code_window>"
        # prepare the target code region
        for line_of_code in sliding_window:
            source_seq += f"<mask>{line_of_code}"
    source_seq += f"<prompt>{prompt}</prompt><prior_edits>"
    source_seq_len = len(tokenizer.encode(source_seq, add_special_tokens=False))
    
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit_seq = prior_edit.formalize_as_prior_edit(beautify=False, label_num=args.label_num)
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        source_seq += prior_edit_seq
        source_seq_len += prior_edit_seq_len
        if source_seq_len + prior_edit_seq_len > 512 - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    source_seq += "</prior_edits>"
    
    return source_seq

def CoEdPilot_formalize_locator_input(sliding_window: list[str], prompt: str, prior_edits: list[CodeWindow], 
                                    tokenizer: RobertaTokenizer, args: argparse.Namespace) -> str:
    source_seq = ""
    # prepare the target code region
    for line_of_code in sliding_window:
        source_seq += f"{tokenizer.mask_token}" + line_of_code
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    common_seq = f"</s>{truncated_prompt}</s>"
    # prepare the prior edits region
    for prior_edit in prior_edits:
        # seperate prefix context, before edit and suffix context from prior edit
        remove_txt = prior_edit.before_edit_region(split_by_line=False, allow_fuzzy=False)
        add_txt = prior_edit.after_edit_region(split_by_line=False)
        if remove_txt == "":
            prior_edit_seq = f"add {add_txt}</s>"
        else:
            prior_edit_seq = f"remove {remove_txt} add {add_txt}</s>"
        common_seq += prior_edit_seq
    source_seq += common_seq

    return source_seq

def locator_predict(locator, locator_tokenizer, dataloader, args, flatten=True):
    """
    Predict the edit operations for each line of code
    
    Args:
        locator: the locator model
        locator_tokenizer: the tokenizer of locator model
        dataloader: the dataloader of locator model
        args: the args object
        
    Returns:
        all_preds: list[list[str]], the predicted edit operations for each line of code
        all_confidences: list[list[float]], the confidence scores for each predicted edit operation
    """
    label_thresholds = {
        "TRACE": {
            "insert_threshold": 0.90,
            "replace_threshold": 0.5,
            "delete_threshold": 0.97,
            "block-split_threshold": 0.97,
        },
        "PlainSemantics": {
            "insert_threshold": 0.98,
            "replace_threshold": 0.96,
        },
        "CoEdPilot": {
            "insert_threshold": 0.95,
            "replace_threshold": 0.90,
        }
    }
    label_thresholds["TRACE-wo-Invoker"] = label_thresholds["TRACE"]
    label_thresholds["EnrichedSemantics"] = label_thresholds["TRACE"]
    all_preds = []
    all_confidences = []
    if args.system in ["TRACE", "TRACE-wo-Invoker", "EnrichedSemantics"]:
        insert_threshold = label_thresholds[args.system]["insert_threshold"]
        replace_threshold = label_thresholds[args.system]["replace_threshold"]
        delete_threshold = label_thresholds[args.system]["delete_threshold"]
        block_split_threshold = label_thresholds[args.system]["block-split_threshold"]
        pure_locator_runtime = 0
        for batch in dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask = batch
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                lm_logits = locator(source_ids=source_ids,source_mask=source_mask, train=False)
                lm_logits = torch.nn.functional.softmax(lm_logits, dim=-1)
                torch.cuda.synchronize()
                one_batch_time = time.time() - start
                pure_locator_runtime += one_batch_time
                
            # extract masked edit operations
            for i in range(lm_logits.shape[0]): # for sample within batch
                    batch_preds = []
                    batch_confidences = []
                    for j in range(lm_logits.shape[1]): # for every token
                        if source_ids[i][j] == locator.inline_mask_id or source_ids[i][j] == locator.inter_mask_id: # if is masked
                            pred_label = locator_tokenizer.decode(torch.argmax(lm_logits[i][j]),clean_up_tokenization_spaces=False)
                            if not pred_label.startswith("<") or not pred_label.endswith(">"):
                                pred_label = f"<{pred_label}>"
                            confidence = torch.max(lm_logits[i][j]).item() # Get the confidence value (0-1)
                            if pred_label == "<insert>" and confidence < insert_threshold: # debug
                                pred_label = "<null>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<null>")].item()
                            elif pred_label == "<replace>" and confidence < replace_threshold: # debug
                                pred_label = "<keep>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<keep>")].item()
                            elif pred_label == "<delete>" and confidence < delete_threshold: # debug
                                pred_label = "<keep>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<keep>")].item()
                            elif pred_label == "<block-split>" and confidence < block_split_threshold: #debug
                                pred_label = "<null>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("<null>")].item()
                            batch_preds.append(pred_label)
                            batch_confidences.append(confidence)
                    all_preds.append(batch_preds)
                    all_confidences.append(batch_confidences)
        
        # all_preds and all_confidences are list[list[str]] and list[list[float]]
        if flatten:
            # We need to: 
            # 1. Flatten to list[str] and list[float]
            # 2. Resolve the conflict between the first & last inter-line label between 2 adjacent code windows
            all_inter_predictions = []
            all_inline_predictions = []
            all_inter_confidences = []
            all_inline_confidences = []
            for preds, confidence in zip(all_preds, all_confidences):
                inter_preds = [preds[i] for i in range(0, len(preds), 2)]
                inline_preds = [preds[i] for i in range(1, len(preds), 2)]
                
                inter_conf = [confidence[i] for i in range(0, len(confidence), 2)]
                inline_conf = [confidence[i] for i in range(1, len(confidence), 2)]
                
                all_inline_predictions.extend(inline_preds)
                all_inline_confidences.extend(inline_conf)
                
                if len(all_inter_predictions) != 0:
                    # compare the last in all_inter_labels and the first in inter_preds
                    if all_inter_confidences[-1] <= inter_conf[0]:
                        # pop the last of all_inter_labels and extend the new
                        all_inter_predictions.pop()
                        all_inter_confidences.pop()
                        all_inter_predictions.extend(inter_preds)
                        all_inter_confidences.extend(inter_conf)
                    else:
                        # pop the first of inter_preds and extend the new
                        inter_preds.pop(0)
                        inter_conf.pop(0)
                        all_inter_predictions.extend(inter_preds)
                        all_inter_confidences.extend(inter_conf)
                else:
                    all_inter_predictions.extend(inter_preds)
                    all_inter_confidences.extend(inter_conf)
            assert len(all_inter_predictions) == len(all_inter_confidences)
            assert len(all_inline_predictions) == len(all_inline_confidences)
            assert len(all_inter_predictions) - 1 == len(all_inline_predictions)

            return {
                "inline_predictions": all_inline_predictions,
                "inline_confidences": all_inline_confidences,
                "inter_predictions": all_inter_predictions,
                "inter_confidences": all_inter_confidences,
                "inline_service": ["normal"] * len(all_inline_predictions),
                "inter_service": ["normal"] * (len(all_inter_predictions) + 1)
            }, pure_locator_runtime
        else:
            return all_preds, all_confidences
    
    elif args.system == "PlainSemantics":
        insert_threshold = label_thresholds[args.system]["insert_threshold"]
        replace_threshold = label_thresholds[args.system]["replace_threshold"]
        pure_locator_runtime = 0
        for batch in dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask = batch                  
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                lm_logits = locator(source_ids=source_ids,source_mask=source_mask, train=False)
                lm_logits = torch.nn.functional.softmax(lm_logits, dim=-1)
                torch.cuda.synchronize()
                one_batch_time = time.time() - start
                pure_locator_runtime += one_batch_time

                # extract masked edit operations
                for i in range(lm_logits.shape[0]): # for sample within batch
                    batch_preds = []
                    batch_confidences = []
                    for j in range(lm_logits.shape[1]): # for every token
                        if source_ids[i][j]==locator_tokenizer.mask_token_id: # if is masked
                            pred_label = locator_tokenizer.decode(torch.argmax(lm_logits[i][j]),clean_up_tokenization_spaces=False)
                            if not pred_label.startswith("<") or not pred_label.endswith(">"):
                                pred_label = f"<{pred_label}>"
                            confidence = torch.max(lm_logits[i][j]).item() # Get the confidence value (0-1)
                            if pred_label == "<insert>" and confidence < insert_threshold:
                                pred_label = "<keep>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("keep")].item()
                            elif pred_label == "<replace>" and confidence < replace_threshold:
                                pred_label = "<keep>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("keep")].item()
                            batch_preds.append(pred_label)
                            batch_confidences.append(confidence)
                    all_preds.append(batch_preds)
                    all_confidences.append(batch_confidences)
        
        # all_preds and all_confidences are list[list[str]] and list[list[float]]
        # We need to: 
        # 1. Flatten to list[str] and list[float]
        all_inline_predictions = []
        all_inline_confidences = []
        for preds, confidence in zip(all_preds, all_confidences):
            all_inline_predictions.extend(preds)
            all_inline_confidences.extend(confidence)
        return {
            "inline_predictions": all_inline_predictions,
            "inline_confidences": all_inline_confidences
        }, pure_locator_runtime
        
    elif args.system == "CodeCloneDetector":
        raise ValueError("CodeCloneDetector is not suppose to reach this branch")
    
    elif args.system == "CoEdPilot":
        insert_threshold = label_thresholds[args.system]["insert_threshold"]
        replace_threshold = label_thresholds[args.system]["replace_threshold"]
        
        pure_locator_runtime = 0
        for batch in dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids,source_mask = batch                  
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                lm_logits = locator(source_ids=source_ids,source_mask=source_mask, train=False)
                lm_logits = torch.nn.functional.softmax(lm_logits, dim=-1)
                torch.cuda.synchronize()
                one_batch_time = time.time() - start
                pure_locator_runtime += one_batch_time

                # extract masked edit operations
                for i in range(lm_logits.shape[0]): # for sample within batch
                    batch_preds = []
                    batch_confidences = []
                    for j in range(lm_logits.shape[1]): # for every token
                        if source_ids[i][j]==locator_tokenizer.mask_token_id: # if is masked
                            pred_label = locator_tokenizer.decode(torch.argmax(lm_logits[i][j]),clean_up_tokenization_spaces=False)
                            if not pred_label.startswith("<") or not pred_label.endswith(">"):
                                pred_label = f"<{pred_label}>"
                            confidence = torch.max(lm_logits[i][j]).item() # Get the confidence value (0-1)
                            if pred_label == "<add>" and confidence < insert_threshold:
                                pred_label = "<keep>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("keep")].item()
                            elif pred_label == "<replace>" and confidence < replace_threshold:
                                pred_label = "<keep>"
                                confidence = lm_logits[i][j][locator_tokenizer.convert_tokens_to_ids("keep")].item()
                            batch_preds.append(pred_label)
                            batch_confidences.append(confidence)
                    all_preds.append(batch_preds)
                    all_confidences.append(batch_confidences)
        # all_preds and all_confidences are list[list[str]] and list[list[float]]
        # We need to: 
        # 1. Flatten to list[str] and list[float]
        all_inline_predictions = []
        all_inline_confidences = []
        for preds, confidence in zip(all_preds, all_confidences):
            all_inline_predictions.extend(preds)
            all_inline_confidences.extend(confidence)
        return {
            "inline_predictions": all_inline_predictions,
            "inline_confidences": all_inline_confidences
        }, pure_locator_runtime
    
    return all_preds, all_confidences

def split_file_into_windows(content, tokenizer):
    """
    Split the file into windows.
    """
    sliding_windows = []
    start_line_idx = 0
    window_length = 10 # default window length is 10
    while True:
        if window_length == 0:
            # code at current start line is too long to fit into one window
            code = content[start_line_idx]
            # 1 token ~= 4 char, expect to have around 256 tokens
            truncated_code = code[:1024]
            sliding_windows.append([truncated_code])
            start_line_idx += 1
            window_length = 10
        window_length = min(window_length, len(content) - start_line_idx)
        if window_length <= 0:
            break
        current_window = content[start_line_idx:start_line_idx+window_length]
        # count token num
        current_window_str = "".join(current_window)
        current_token_num = len(tokenizer.tokenize(current_window_str))
        """
        Input must have enough space to store the special tokens for code window, including:
        2: <code_window>, </code_window>
        l: <inline-mask> for l lines of code
        l+1: <inter-mask> for l + 1 spaces between lines of code
        """
        redundancy = 2 + 2 * len(current_window) + 1
        if redundancy + current_token_num >= 512:
            # current window is too long, reduce the length and try again
            window_length -= 1 
            continue
        else:
            sliding_windows.append(current_window.copy())
            start_line_idx += window_length
            window_length = 10 # reset to default length
            
    return sliding_windows