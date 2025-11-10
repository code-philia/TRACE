import torch
import logging
import argparse
import warnings
import torch.nn as nn

from invoker_utils import *
from invoker import *
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    if True:
        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="model type: e.g. roberta, codet5" )  
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model: e.g. roberta-base" )   
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--load_model_path", default=None, type=str,
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
        
        parser.add_argument("--batch_size", default=8, type=int, required=True,
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
        parser.add_argument("--meta_dataset_path", type=str, default="meta_dataset.json",
                            help="Path to meta dataset")
        parser.add_argument("--dataset_dir", type=str, default="dataset",
                            help="Path to dataset")
        
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
    
    invoker, tokenizer = load_model_tokenizer(args, logger)
    invoker.to(args.device)
    
    if args.do_train:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in invoker.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
            {'params': [p for n, p in invoker.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        dataloader_size = get_dataloader_size(args)
        t_total = dataloader_size // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
        
        # training loop
        nb_tr_examples, nb_tr_steps,global_step,best_f1 = 0,0,0,0
        recording_variables = [nb_tr_examples, nb_tr_steps, global_step, best_f1]
        
        train_data = load_and_tokenize_data(args.train_filename, tokenizer, args, logger)
        valid_data = load_and_tokenize_data(args.dev_filename, tokenizer, args, logger)
            
        for epoch in range(args.num_train_epochs):
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
                valid_sampler = RandomSampler(valid_data)
            else:
                train_sampler = DistributedSampler(train_data)
                valid_sampler = DistributedSampler(valid_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size//args.gradient_accumulation_steps)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
            
            logger.info("***** Training Invoker *****")
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num epoch = %d", args.num_train_epochs)

            recording_variables = train_invoker(args, epoch, invoker, train_dataloader, \
                valid_dataloader, recording_variables, optimizer, scheduler, logger)
    
    if args.do_eval:
        logger.info("***** Evaluate Invoker *****")
        
        dev_data = load_and_tokenize_data(args.dev_filename, tokenizer, args, logger)
        
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        marco_f1, probabilities, prediction, gold = eval_invoker(invoker, dev_dataloader, args)
        
        logger.info("***** Dev Result *****")
        logger.info("  F1 = %s", marco_f1)
        
        np.save(os.path.join(args.output_dir, "dev_probs.npy"), probabilities)
        np.save(os.path.join(args.output_dir, "dev_preds.npy"), prediction)
        np.save(os.path.join(args.output_dir, "dev_golds.npy"), gold)
    
    if args.do_test:
        logger.info("***** Test Invoker *****")
        
        test_data = load_and_tokenize_data(args.test_filename, tokenizer, args, logger)
        
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        marco_f1, probabilities, prediction, gold = eval_invoker(invoker, test_dataloader, args)

        print(classification_report(gold, prediction, output_dict=False, digits=4))
        logger.info("***** Test Result *****")
        logger.info("  F1 = %s", marco_f1)
        
        np.save(os.path.join(args.output_dir, "test_probs.npy"), probabilities)
        np.save(os.path.join(args.output_dir, "test_preds.npy"), prediction)
        np.save(os.path.join(args.output_dir, "test_golds.npy"), gold)
        
        
if __name__ == "__main__":
    main()