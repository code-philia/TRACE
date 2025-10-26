import os
import bleu
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
import multiprocessing

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from code_window import CodeWindow
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true', help="Whether to evaluate bleu on dev set.")

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_generator_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    # parser.add_argument("--load_dep_model_path", default=None, type=str,
    #                     help="Path to trained model: Should contain the .bin files")
    # parser.add_argument("--load_estimator_model_path", default=None, type=str,
    #                     help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--estimator_batch_size", default=20, type=int,
                        help="Batch size per GPU/CPU for estimator.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Whether to run in debug mode")
    parser.add_argument('--debug_size', type=int, default=20,
                        help="Size of the debug dataset")
    parser.add_argument('--select_method', type=str, choices=["random", "bm25"], required=True)
    parser.add_argument('--label_num', type=int, default=6)
    args = parser.parse_args()

    return args

def formalize_generator_input(sliding_window: dict, prompt: str, 
                            prior_edits: list[dict], tokenizer: RobertaTokenizer, args: argparse.Namespace) -> tuple[str, str]:
    external_tool_feedback = sliding_window["external_tool_feedback"]
    sliding_window = CodeWindow(sliding_window, "hunk")
    common_seq = f"<feedback>{external_tool_feedback}</feedback>"+ sliding_window.formalize_as_generator_target_window(beautify=False, label_num=args.label_num)
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    common_seq += f"</code_window><prompt>{truncated_prompt}</prompt><prior_edits>"
    common_seq_len = len(tokenizer.encode(common_seq, add_special_tokens=False))
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit = CodeWindow(prior_edit, "hunk")
        prior_edit_seq = prior_edit.formalize_as_prior_edit(beautify=False, label_num=args.label_num)
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        common_seq += prior_edit_seq
        common_seq_len += prior_edit_seq_len
        if common_seq_len + prior_edit_seq_len > args.max_source_length - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    common_seq += "</prior_edits>"
    target_seq = "".join(sliding_window.after_edit)

    return common_seq, target_seq

def formalize_generator_input_CoEdPilot(sliding_window: dict, prompt: str, 
                            prior_edits: list[dict], tokenizer: RobertaTokenizer, args: argparse.Namespace) -> tuple[str, str]:
    sliding_window = CodeWindow(sliding_window, "hunk")
    common_seq = sliding_window.formalize_as_generator_target_window(beautify=False, label_num=args.label_num)
    common_seq = common_seq.replace("<keep>", "keep").replace("<replace>", "replace").replace("<insert>", "add").replace("<code_window>","").replace("</code_window>","")
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    common_seq += f"</s>{truncated_prompt}</s>"
    common_seq_len = len(tokenizer.encode(common_seq, add_special_tokens=False))
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit = CodeWindow(prior_edit, "hunk")
        prior_edit_seq = f"remove{prior_edit.before_edit_region(split_by_line=False)}</s> add {prior_edit.after_edit_region(split_by_line=False)}</s>"
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        common_seq += prior_edit_seq
        common_seq_len += prior_edit_seq_len
        if common_seq_len + prior_edit_seq_len > args.max_source_length - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    target_seq = "".join(sliding_window.after_edit)

    return common_seq, target_seq

def get_code_distance(code1: dict, code2: dict) -> float:
    """
    Func:
        Get the code distance between 2 code snippet
    Args:
        code1: dict, must have key "file_path" and "edit_start_line_idx"
        code2: dict, must have key "file_path" and "edit_start_line_idx"
    Return:
        distance: float, in [0, 1]
    """
    if code1.file_path != code2.file_path:
        return 0
    
    return max(0, 1 - abs(code1.edit_start_line_idx - code2.edit_start_line_idx) / 50)

def select_hunk(tgt_hunk: dict, other_hunks: list[dict], args: argparse.Namespace, selector_model_set) -> list[dict]:
    """
    Func: 
        Given a target hunk and a list of other hunks, select the prior edits from the other hunks
    Args:
        tgt_hunk: dict, the target hunk
        other_hunks: list[dict], the other hunks
        estimator: Estimator, the prior edit estimator
        estimator_tokenizer: RobertaTokenizer, the tokenizer for estimator
        dependency_tokenizer: RobertaTokenizer, the tokenizer for dependency
        args: argparse.Namespace, the arguments
    Return:
        prior_edits: list[dict], the prior edits
    """
    non_overlap_hunks = [CodeWindow(hunk, "hunk") for hunk in other_hunks]
    tgt_hunk_obj = CodeWindow(tgt_hunk, "hunk")
    if args.select_method == "bm25":
        tokenized_corpus = ["".join(hunk.before_edit_window()+hunk.after_edit_region()).split() for hunk in non_overlap_hunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = tgt_hunk_obj.before_edit_window(split_by_line=False).split()
        retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3)
        retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
        prior_edits = [other_hunks[idx] for idx in retrieved_index] # get corresponding hunk id
        assert len(prior_edits) <= 3
    elif args.select_method == "random":
        raise NotImplementedError("Not implemented yet")
    elif args.select_method == "tfidf":
        raise NotImplementedError("Not implemented yet")

    return prior_edits
    
def load_data(args: argparse.Namespace, filename: str, generator_tokenizer: RobertaTokenizer, 
              split_tag, only_src=False, selector_model_set=None):
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    examples=[]
    ids = []
    hunk_selected_dataset = []
    with open(filename, "r") as f:
        dataset = json.load(f)
    for idx, (commit_url, commit) in enumerate(tqdm(dataset.items(), desc=f"Load {split_tag} data & select prior edits")):
        hunks = commit['hunks']
        commit_msg = commit['commit_msg']
        for hunk in hunks:
            other_hunks = [h for h in hunks if h != hunk]    
            selected_hunks = select_hunk(hunk, other_hunks, args, selector_model_set)
            hunk_selected_dataset.append({
                "hunk": hunk,
                "commit_msg": commit_msg,
                "selected_hunks": selected_hunks,
                "commit_url": commit_url,
            })
            # shuffle the other hunks
            source_seq, target_seq = formalize_generator_input(hunk, commit_msg, selected_hunks, generator_tokenizer, args)
            encoded_source_seq = generator_tokenizer(source_seq, padding="max_length", truncation=True, max_length=args.max_source_length)
            source_ids = encoded_source_seq["input_ids"]

            encoded_target_seq = generator_tokenizer(target_seq, padding="max_length", truncation=True, max_length=args.max_target_length)
            target_ids = encoded_target_seq["input_ids"]
            ids.append((source_ids, target_ids))
            examples.append({
                "source":source_seq, 
                "target":target_seq
            })
        if args.debug_mode and args.debug_size == idx:
            break
    
    # save hunk_selected_dataset to jsonl file
    if not os.path.exists(os.path.join(args.output_dir, "fixed_dataset")):
        os.makedirs(os.path.join(args.output_dir, "fixed_dataset"))
    with open(os.path.join(args.output_dir, "fixed_dataset", f"{split_tag}_hunk_selected.jsonl"), "w") as f:
        for item in hunk_selected_dataset:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(args.output_dir, "fixed_dataset", f"{split_tag}_input.jsonl"), "w") as f:
        for item in examples:
            f.write(json.dumps(item) + "\n")
    if split_tag == 'test' or only_src:
        all_source_ids = torch.tensor([id_pair[0] for id_pair in ids], dtype=torch.long)  
        data = TensorDataset(all_source_ids)
    else:
        all_source_ids = torch.tensor([id_pair[0] for id_pair in ids], dtype=torch.long)
        all_target_ids = torch.tensor([id_pair[1] for id_pair in ids], dtype=torch.long) 
        data = TensorDataset(all_source_ids,all_target_ids)
    
    return examples, data

def get_filenames(data_root, lang):
    data_dir = os.path.join(data_root, lang)
    train_fn = os.path.join(data_dir, 'train.json')
    dev_fn = os.path.join(data_dir, 'dev.json')
    test_fn = os.path.join(data_dir, 'test.json')
    return train_fn, dev_fn, test_fn

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)

def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
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
                