import os
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from code_window import CodeWindow
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import tempfile
import re

def make_locator_dataset(dataset_path: str, locator_tokenizer: RobertaTokenizer,
                         args: argparse.Namespace, logger: logging.Logger , epoch: int = 1) -> tuple[TensorDataset, list[dict]]:
    """
    Func:
        Select most relevant hunk as prior edit.
        Make locator dataset for locator training.
    Args:
        dataset_path: the path of raw dataset from ../crawl/new_dataset/{lang}_dataset.json
        locator_tokenizer: the tokenizer for locator
        args: argparse.Namespace, the arguments
        logger: logging.Logger, the logger
        epoch: int, the epoch of training, if 0, then select prior edit by random
    """
    with open(dataset_path, "r") as f:
        raw_dataset = json_to_object(json.load(f))
    
    locator_dataset_source_ids = []
    locator_dataset_source_masks = []
    locator_dataset_target_ids = []
    raw_locator_dataset = []
    prior_edit_nums = []
    for idx, (commit_url, commit) in enumerate(tqdm(raw_dataset.items(), desc="Finding relevant prior edits")): # for each commit
        commit_msg = commit["commit_msg"]
        hunks = commit["hunks"]
        sliding_windows = commit["sliding_windows"]
        
        for sliding_window in sliding_windows:
            if args.select_method == "random":
                non_overlap_hunks = [hunk for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids]
                prior_edits = random.sample(non_overlap_hunks, min(len(non_overlap_hunks), 3))
            elif args.select_method == "bm25":
                non_overlap_hunks = [hunk for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids]
                choosen_hunk_ids = [hunk.id for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids] # index to hunk id
                tokenized_corpus = [locator_tokenizer.tokenize("".join(hunk.before_edit_region()+hunk.after_edit_region())) for hunk in non_overlap_hunks]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = locator_tokenizer.tokenize("".join(sliding_window.code_window))
                retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3) 
                retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
                prior_edit_id = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk id
                prior_edits = []
                for id in prior_edit_id: # preserve the order
                    prior_edits.append([hunk for hunk in hunks if hunk.id == id][0])
            elif args.select_method == "tfidf":
                non_overlap_hunks = [hunk for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids]
                choosen_hunk_ids = [hunk.id for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids]
                corpus = ["".join(hunk.before_edit_window()+hunk.after_edit_window()) for hunk in non_overlap_hunks]
                # init a TF-IDF Vectorizer
                vectorizer = TfidfVectorizer()
                # fit and transform the corpus 
                tfidf_matrix = vectorizer.fit_transform(corpus)
                # construct query
                query = "".join(sliding_window.code_window)
                query_vector = vectorizer.transform([query])
                # compute the cos similiarity between query and document
                cosine_similarities = np.dot(query_vector, tfidf_matrix.T).toarray().flatten()
                # get the top-3 similar documents' index 
                retrieved_index = cosine_similarities.argsort()[-3:][::-1]
                prior_edit_id = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk id
                prior_edits = []
                for id in prior_edit_id: # preserve the order
                    prior_edits.append([hunk for hunk in hunks if hunk.id == id][0])
            # for picked prior edits, combine with sliding window, turn into locator data sample
            source_seq, target_seq = formalize_locator_input(sliding_window, commit_msg, prior_edits, locator_tokenizer, args)
            if args.label_num == 6:
                raw_locator_dataset.append({
                    "source_seq": source_seq,
                    "inline_labels": sliding_window.inline_labels,
                    "inter_labels": sliding_window.inter_labels,
                    "commit_url": commit_url
                })
            elif args.label_num == 3:
                raw_locator_dataset.append({
                    "source_seq": source_seq,
                    "labels": label_conversion(sliding_window.inline_labels, sliding_window.inter_labels),
                    "commit_url": commit_url
                })
            source_ids, source_mask, target_ids = mlm_tokenization(source_seq, sliding_window, locator_tokenizer, args)
            if source_ids == None:
                # if returned none, the target code window is longer than allowed input length
                continue
            token_match_assertion(locator_tokenizer, source_ids, target_ids, args)
            prior_edit_num = torch.eq(source_ids, locator_tokenizer.convert_tokens_to_ids("<edit>")).sum().item()
            prior_edit_nums.append(prior_edit_num) # count the number of prior edits that can be encoded
            locator_dataset_source_ids.append(source_ids)
            locator_dataset_source_masks.append(source_mask)
            locator_dataset_target_ids.append(target_ids)
        if args.debug_mode and idx + 1 == args.debug_size: # debug mode only use 10 commit
            break
    
    logger.info(f"Average number of prior edits: {sum(prior_edit_nums) / len(prior_edit_nums)}")
    locator_dataset_source_ids = torch.stack(locator_dataset_source_ids, dim=0)
    locator_dataset_source_masks = torch.stack(locator_dataset_source_masks, dim=0)
    locator_dataset_target_ids = torch.stack(locator_dataset_target_ids, dim=0)
    dataset = TensorDataset(locator_dataset_source_ids, locator_dataset_source_masks, locator_dataset_target_ids)
    
    return dataset, raw_locator_dataset
           
def formalize_locator_input(sliding_window: dict, prompt: str, 
                            prior_edits: list[dict], tokenizer: RobertaTokenizer, args: argparse.Namespace) -> tuple[str, str]:
    source_seq, target_seq = sliding_window.formalize_as_locator_target_window(beautify=False, label_num=args.label_num)
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    common_seq = f"<prompt>{truncated_prompt}</prompt><prior_edits>"
    # get the # of tokens in common_seq
    common_seq_len = len(tokenizer.encode(common_seq, add_special_tokens=False))
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit_seq = prior_edit.formalize_as_prior_edit(beautify=False, label_num=args.label_num)
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        common_seq += prior_edit_seq
        common_seq_len += prior_edit_seq_len
        if common_seq_len + prior_edit_seq_len > args.max_source_length - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    common_seq += "</prior_edits>"
    source_seq += common_seq
    target_seq += common_seq

    return source_seq, target_seq

def mlm_tokenization(source_seq: str, sliding_window: dict, tokenizer: RobertaTokenizer, 
                     args: argparse.Namespace) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    # encode the source_seq
    encoded_source_seq = tokenizer(source_seq, padding="max_length", truncation=True, max_length=args.max_source_length)
    source_ids = encoded_source_seq["input_ids"]
    source_mask = encoded_source_seq["attention_mask"]

    # replace mask with edit operation label token
    target_ids = source_ids.copy()
    inline_label_idx = 0
    inter_label_idx = 0
    # print(source_seq)
    for idx, token_id in enumerate(source_ids):
        if args.label_num == 6:
            if token_id == tokenizer.mask_token_id:
                target_id = tokenizer.convert_tokens_to_ids(f"<{sliding_window.inline_labels[inline_label_idx]}>")
                assert target_id in [tokenizer.convert_tokens_to_ids("<replace>"), tokenizer.convert_tokens_to_ids("<delete>"), tokenizer.convert_tokens_to_ids("<keep>")]
                target_ids[idx] = target_id
                inline_label_idx += 1
            elif token_id == tokenizer.convert_tokens_to_ids("<inter-mask>"):
                target_id = tokenizer.convert_tokens_to_ids(f"<{sliding_window.inter_labels[inter_label_idx]}>")
                assert target_id in [tokenizer.convert_tokens_to_ids("<null>"), tokenizer.convert_tokens_to_ids("<insert>"), tokenizer.convert_tokens_to_ids("<block-split>")]
                target_ids[idx] = target_id
                inter_label_idx += 1
        elif args.label_num == 3:
            labels = label_conversion(sliding_window.inline_labels, sliding_window.inter_labels)
            if token_id == tokenizer.mask_token_id:
                target_id = tokenizer.convert_tokens_to_ids(f"<{labels[inline_label_idx]}>")
                assert target_id in [tokenizer.convert_tokens_to_ids("<replace>"), tokenizer.convert_tokens_to_ids("<insert>"), tokenizer.convert_tokens_to_ids("<keep>")]
                target_ids[idx] = target_id
                inline_label_idx += 1
    if args.label_num == 6:
        try:
            assert inline_label_idx == len(sliding_window.inline_labels)
            assert inter_label_idx == len(sliding_window.inter_labels)
        except:
            # In case the code window is too long to encode
            return None, None, None
    elif args.label_num == 3:
        try:
            assert inline_label_idx == len(sliding_window.inline_labels)
        except:
            return None, None, None
    return torch.tensor(source_ids), torch.tensor(source_mask), torch.tensor(target_ids)
      
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_locator_dataloader_size(args: argparse.Namespace) -> int:
    with open(args.train_filename, "r") as f:
        raw_dataset = json.load(f)

    sliding_window_size = 0
    for idx, (commit_url, commit) in enumerate(raw_dataset.items()):
        sliding_window_size += len(commit["sliding_windows"])
        if args.debug_mode and idx + 1 == args.debug_size:
            break
    return sliding_window_size // args.locator_batch_size

def json_to_object(json_dict: dict) -> dict:
    """
    Func:
        Convert a json dict to dict of object
    Args:
        json_dict: dict
    Return:
        obj: object
    """
    object_dict = {}
    for commit_url, commit_content in json_dict.items():
        object_dict[commit_url] = {
            "commit_msg": commit_content["commit_msg"],
            "hunks": [],
            "sliding_windows": []
        }
        for hunk in commit_content["hunks"]:
            object_dict[commit_url]["hunks"].append(CodeWindow(hunk, "hunk"))
        for sliding_window in commit_content["sliding_windows"]:
            object_dict[commit_url]["sliding_windows"].append(CodeWindow(sliding_window, "sliding_window"))
    
    return object_dict

def token_match_assertion(tokenizer, source_ids, target_ids, args):
    """
    Func:
        Assert the <mask> & <inter-mask> in source_ids and target_ids are matched with corresponding labels
    """
    for source_id, target_id in zip(source_ids, target_ids):
        if args.label_num == 6:
            if source_id == tokenizer.mask_token_id:
                assert target_id in [tokenizer.convert_tokens_to_ids("<replace>"), tokenizer.convert_tokens_to_ids("<delete>"), tokenizer.convert_tokens_to_ids("<keep>")]
            if source_id == tokenizer.convert_tokens_to_ids("<inter-mask>"):
                assert target_id in [tokenizer.convert_tokens_to_ids("<null>"), tokenizer.convert_tokens_to_ids("<insert>"), tokenizer.convert_tokens_to_ids("<block-split>")]
        elif args.label_num == 3:
            if source_id == tokenizer.mask_token_id:
                assert target_id in [tokenizer.convert_tokens_to_ids("<replace>"), tokenizer.convert_tokens_to_ids("<insert>"), tokenizer.convert_tokens_to_ids("<keep>")]
            
def label_conversion(inline_labels: list[str], inter_labels: list[str]) -> list[str]:
    """
    Func:
        Given the fine grain label of new models, convert them to old labels
    Args:   
        inline_labels: list[str], have label: keep, replace, delete
        inter_labels: list[str], have label: null, insert, block-split
    Return:
        old_labels: list[str]
    """
    assert len(inline_labels) + 1 == len(inter_labels)
    # rule 1: block-split can be ignored
    inter_labels = ["null" if x == "block-split" else x for x in inter_labels]
    
    # rule 2: delete is a part of replace
    inline_labels = ["replace" if x == "delete" else x for x in inline_labels]
    
    # rule 3: old labels can't handle insert at the beginning of code window
    inter_labels = inter_labels[1:]
    
    old_labels = []
    # rule 4: now inter_label  should only have null & insert
    #             inline_label should only have keep & replace
    for inter_label, inline_label in zip(inter_labels, inline_labels):
        if inter_label == "null":
            old_labels.append(inline_label)
        else: # inter_label == "insert"
            if inline_label == "keep":
                old_labels.append("insert")
            else:
                old_labels.append("replace")
                
    assert len(old_labels) == len(inline_labels)
    return old_labels

def word_diff(str1, str2):
    # create tmp file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file1, tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file2:
        temp_file1.write(str1)
        temp_file2.write(str2)
        
        temp_file1_name = temp_file1.name
        temp_file2_name = temp_file2.name
    
    # use git diff to compare 2 tmp file
    result = subprocess.run(['git', 'diff', '--word-diff', '--no-index', temp_file1_name, temp_file2_name], capture_output=True, text=True)
    
    # delete tmp file
    subprocess.run(['rm', temp_file1_name, temp_file2_name])
    
    # parse diff output 
    pattern = r'^@@.*?@@$\n(.*)'

    # search of regex matching
    matches = re.findall(pattern, result.stdout, re.DOTALL | re.MULTILINE)

    try:
        git_diff = matches[0]
    except:
        return None
    
    add_pattern = r'\{\+([\s\S]*?)\+\}'
    add_matches = re.findall(add_pattern, git_diff)
    del_pattern = r'\[-([\s\S]*?)-\]'
    del_matches = re.findall(del_pattern, git_diff)
    
    add_len = sum([len(match.split()) for match in add_matches])
    del_len = sum([len(match.split()) for match in del_matches])
    
    modify_len = (add_len + del_len) / 2
    sentence_len = (len(str1.split()) + len(str2.split())) / 2
    modify_percent = modify_len / sentence_len
    
    if modify_percent < 0.20:
        keywords = []
        for match in add_matches:
            keywords.extend(match.split())
        for match in del_matches:
            keywords.extend(match.split())
        return keywords
    else:
        return None

class CustomTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(CustomTfidfVectorizer, self).build_analyzer()
        return lambda doc: [w for w in analyzer(doc)]
    
    def _tfidf(self, X, custom_weights, norm='l2'):
        if not self.use_idf:
            return X
        idf = self._idf_diag
        tfidf = X.dot(idf)
        tfidf = tfidf.toarray()
        
        feature_names = np.array(self.get_feature_names_out())
        for word in custom_weights:
            if word in feature_names:
                idx = np.where(feature_names == word)[0][0]
                tfidf[:, idx] *= 2.0
        
        return tfidf
  