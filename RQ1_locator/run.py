import argparse
import logging
import torch
import torch.nn as nn

from utils import *
from convert_label import label_conversion
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import classification_report, fbeta_score
from locator import Locator ,train_locator,evaluate_locator,locator_loss_by_sample
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config,T5ForConditionalGeneration)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    if True:
        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type: e.g. roberta")
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model: e.g. roberta-base" )   
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--load_locator_model_path", default=None, type=str,
                            help="Path to trained model: Should contain the .bin files" )    
        ## Other parameters
        parser.add_argument("--train_filename", default=None, type=str, 
                            help="The train filename. Should contain the .jsonl files for this task.")
        parser.add_argument("--dev_filename", default=None, type=str, 
                            help="The dev filename. Should contain the .jsonl files for this task.")
        parser.add_argument("--test_filename", default=None, type=str, 
                            help="The test filename. Should contain the .jsonl files for this task.")  
        
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name") 
        parser.add_argument("--max_source_length", default=64, type=int,
                            help="The maximum total source sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--do_test", action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available") 
        
        parser.add_argument("--locator_batch_size", default=8, type=int, required=True,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=3, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--eval_steps", default=-1, type=int,
                            help="")
        parser.add_argument("--train_steps", default=-1, type=int,
                            help="")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")   
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument("--debug_mode", action="store_true", 
                            help="In debug mode, we only use data in debug_size commits")
        parser.add_argument("--debug_size", type=int, default=10,
                            help="In debug mode, we only use data in debug_size commits")
        parser.add_argument("--lang", default=None, type=str, required=True,
                            help="language of the dataset")
        parser.add_argument("--keep_weight", default=1.0, type=float,
                            help="weight for <keep>")
        parser.add_argument("--delete_weight", default=1.0, required=True, type=float,
                            help="weight for <delete>")
        parser.add_argument("--replace_weight", default=1.0, required=True, type=float,
                            help="weight for <replace>")
        parser.add_argument("--null_weight", default=1.0, type=float,
                            help="weight for <null>")
        parser.add_argument("--insert_weight", default=1.0, required=True, type=float,
                            help="weight for <insert>")
        parser.add_argument("--block_split_weight", default=1.0, required=True, type=float,
                            help="weight for <block-split>")
        parser.add_argument("--select_method", default="bm25", type=str, required=True, choices=["random", "bm25", "tfidf"],
                            help="Method to select the data. Must be one of 'selector', 'random', or 'bm25'.")
        parser.add_argument("--label_num", default=6, type=int, required=True,
                            help="number of labels")
        # print arguments
        args = parser.parse_args()
        logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device

    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    print("model_name_or_path:",args.model_name_or_path)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    locator_config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    locator_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    # build locator model
    if args.model_type == "codet5":
        codeT5 = model_class.from_pretrained(args.model_name_or_path) 
        encoder = codeT5.encoder
    else:
        encoder = model_class.from_pretrained(args.model_name_or_path)
    # add special tokens
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "</insert>","<replace-by>", "</replace-by>"]
                        #   "<block-delete>", "</block-delete>",
                        #   "<block-insert>", "</block-insert>"]
    locator_tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    encoder.resize_token_embeddings(len(locator_tokenizer))
    locator_config.vocab_size = len(locator_tokenizer)
    locator=Locator(encoder=encoder,config=locator_config,
                    inline_mask_id=locator_tokenizer.mask_token_id,
                    inter_mask_id=locator_tokenizer.convert_tokens_to_ids("<inter-mask>"),
                    keep_token_id=locator_tokenizer.convert_tokens_to_ids("<keep>"),
                    delete_token_id=locator_tokenizer.convert_tokens_to_ids("<delete>"),
                    replace_token_id=locator_tokenizer.convert_tokens_to_ids("<replace>"),
                    null_token_id=locator_tokenizer.convert_tokens_to_ids("<null>"),
                    insert_token_id=locator_tokenizer.convert_tokens_to_ids("<insert>"),
                    block_split_token_id=locator_tokenizer.convert_tokens_to_ids("<block-split>"),
                    args=args)

    if args.load_locator_model_path is not None:
        logger.info("reload model from {}".format(args.load_locator_model_path))
        locator.load_state_dict(torch.load(args.load_locator_model_path, weights_only=True))

    locator.to(device)

    if args.do_train:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in locator.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
            {'params': [p for n, p in locator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        dataloader_size = get_locator_dataloader_size(args)
        t_total = dataloader_size // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
        
        # training loop
        nb_tr_examples, nb_tr_steps,global_step,best_ppl = 0,0,0,1e6
        recording_variables = [nb_tr_examples, nb_tr_steps, global_step, best_ppl]

        for epoch in range(args.num_train_epochs):
            logger.info(f"***** Epoch {epoch}, Training locator *****")
            # Step 1: train locator
            # Step 1.1: make locator dataset for locator training
            if epoch == 0: 
                logger.info("***** Finding relevant prior edit for training dataset *****")
                locator_train_data, locator_raw_train_data = make_locator_dataset(args.train_filename, locator_tokenizer, args, logger, epoch)
                logger.info("***** Finding relevant prior edit for validation dataset *****")
                locator_valid_data, locator_raw_valid_data = make_locator_dataset(args.dev_filename, locator_tokenizer, args, logger, epoch)

                if os.path.exists(os.path.join(args.output_dir, "locator_data")) is False:
                    os.makedirs(os.path.join(args.output_dir, "locator_data"))
                with open(os.path.join(args.output_dir, "locator_data", f"train_{epoch}.jsonl"), "w") as f:
                    for data in locator_raw_train_data:
                        f.write(json.dumps(data) + "\n")
                with open(os.path.join(args.output_dir, "locator_data", f"dev_{epoch}.jsonl"), "w") as f:
                    for data in locator_raw_valid_data:
                        f.write(json.dumps(data) + "\n")

            if args.local_rank == -1:
                train_sampler = RandomSampler(locator_train_data)
                valid_sampler = RandomSampler(locator_valid_data)
            else:
                train_sampler = DistributedSampler(locator_train_data)
                valid_sampler = DistributedSampler(locator_valid_data)
            train_dataloader = DataLoader(locator_train_data, sampler=train_sampler, batch_size=args.locator_batch_size//args.gradient_accumulation_steps)
            valid_dataloader = DataLoader(locator_valid_data, sampler=valid_sampler, batch_size=args.locator_batch_size)
            # Step 1.2: train locator
            logger.info("***** Training locator *****")
            logger.info("  Batch size = %d", args.locator_batch_size)
            logger.info("  Num epoch = %d", args.num_train_epochs)

            recording_variables = train_locator(args, epoch, locator, locator_tokenizer, train_dataloader, \
                valid_dataloader, device, recording_variables, optimizer, scheduler, logger)
            
    if args.do_test:
        # evaluate_locator
        logger.info("***** Evaluate Locator*****")
        locator_test_data, locator_raw_test_data = make_locator_dataset(args.test_filename, locator_tokenizer, args, logger)
        logger.info(f"test data size: {len(locator_test_data)}")
        os.makedirs(os.path.join(args.output_dir, "locator_data"), exist_ok=True)
        with open(os.path.join(args.output_dir, "locator_data", f"test_{args.select_method}.jsonl"), "w") as f:
            for data in locator_raw_test_data:
                f.write(json.dumps(data) + "\n")
        test_dataloader = DataLoader(locator_test_data, batch_size=args.locator_batch_size, shuffle=False)
        pred_edit_labels, gold_edit_labels, confidences = evaluate_locator(locator, test_dataloader, device, locator_tokenizer, logger, args)

        with open(os.path.join(args.output_dir, f"test_{args.select_method}.gold"), "w") as f:
            for idx, gold_labels in enumerate(gold_edit_labels):
                seq = " ".join(gold_labels)
                f.write(f"{idx}\t{seq}\n")
        with open(os.path.join(args.output_dir, f"test_{args.select_method}.pred"), "w") as f:
            for idx, pred_labels in enumerate(pred_edit_labels):
                seq = " ".join(pred_labels)
                f.write(f"{idx}\t{seq}\n")
        with open(os.path.join(args.output_dir, f"test_confidence_{args.select_method}.json"), "w") as f:
            json.dump(confidences, f)

        print_result(args.output_dir)

def print_result(output_dir):
    with open(f"{output_dir}/test_bm25.pred", "r") as f:
        preds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    with open(f"{output_dir}/test_bm25.gold", "r") as f:
        golds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    with open(f"{output_dir}/test_confidence_bm25.json", "r") as f:
        confidences = json.load(f)

    # Get the classification report if we convert back to 3 labels:
    converted_preds = []
    converted_golds = []
    converted_confs = []
    for pred, gold, confidence in zip(preds, golds, confidences):
        pred_inter_line = [pred[i] for i in range(0, len(pred), 2)]
        pred_inline = [pred[i] for i in range(1, len(pred), 2)]
        gold_inter_line = [gold[i] for i in range(0, len(gold), 2)]
        gold_inline = [gold[i] for i in range(1, len(gold), 2)]
        
        converted_label, converted_conf = label_conversion(pred_inline, pred_inter_line, confidence)
        converted_preds.extend(converted_label)
        converted_confs.extend(converted_conf)
        converted_golds.extend(label_conversion(gold_inline, gold_inter_line))
    print("==> Classification report (Converted to 3 labels)")
    print(classification_report(converted_golds, converted_preds, digits=4, labels=["<keep>", "<insert>", "<replace>"]), end="")

if __name__ == "__main__":
    main()
