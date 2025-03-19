# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import math
import time
import torch
import logging
import argparse
import multiprocessing

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from generator import build_or_load_gen_model,train_generator,evaluate_generator
from utils import add_args, set_seed, set_dist, get_filenames, get_elapse_time, load_data
from prior_edit_estimator import load_estimator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    _, generator, generator_tokenizer = build_or_load_gen_model(args)
    if args.paper_name == "CoEdPilot":
        args.select_method = "selector"
        args.label_num = 3
        args.load_dep_model_path = "../dependency_analyzer/model"
        args.load_estimator_model_path = "fixed_model/estimator.bin"
        estimator, estimator_tokenizer, dependency_tokenizer = load_estimator(args) 
    generator.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        generator = torch.nn.DataParallel(generator)

    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.lang)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        tb_writer = None
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        if args.select_method == "selector":
            selector_model_set = [estimator, estimator_tokenizer, dependency_tokenizer]
            train_examples, train_data = load_data(args, args.train_filename, generator_tokenizer, split_tag='train', selector_model_set=selector_model_set)
            eval_examples, eval_data = load_data(args, args.dev_filename, generator_tokenizer, 'dev', selector_model_set=selector_model_set)
        else:
            train_examples, train_data = load_data(args, args.train_filename, generator_tokenizer, split_tag='train')
            eval_examples, eval_data = load_data(args, args.dev_filename, generator_tokenizer, 'dev')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6 # for early stop

        recording_variables = [global_step, best_bleu_em, best_ppl, not_loss_dec_cnt, not_bleu_em_inc_cnt]

        for epoch in range(args.start_epoch, int(args.num_train_epochs)):
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                        num_workers=4, pin_memory=True)

            # generator train epoch
            recording_variables, early_stop = train_generator(
                args,epoch,generator,train_dataloader,eval_dataloader,
                recording_variables, generator_tokenizer,optimizer,scheduler,
                tb_writer,fa,logger)

            if early_stop == True:
                break


        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        # prepare eval dataloader and examples
        if args.paper_name == "CoEdPilot":
            selector_model_set = [estimator, estimator_tokenizer, dependency_tokenizer]
            eval_examples, eval_data = load_data(args, args.test_filename, generator_tokenizer,'test', only_src=True, selector_model_set=selector_model_set)
        else:
            eval_examples, eval_data = load_data(args, args.test_filename, generator_tokenizer,'test', only_src=True)
        eval_sampler = SequentialSampler(eval_data)
        if args.data_num == -1:
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                        num_workers=4, pin_memory=True)
        else:
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        #eval generator using eval_dataloader
        eval_results = evaluate_generator(args,generator,eval_dataloader,eval_examples,generator_tokenizer,fa,logger)

        if args.res_fn:
            for result_info in eval_results: # [test bin file, result_str, test_metric1, ......]
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), result_info[0]))
                    f.write(result_info[1])


    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
