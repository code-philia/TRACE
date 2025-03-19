import os
import json
import torch
import argparse
import numpy as np
import logging
import bleu
from transformers import (RobertaTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer)

from tqdm import tqdm
from utils import load_data
from torch.nn import DataParallel
from transformers import RobertaTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'t5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path)
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "<replace-by>", "</replace-by>",
                          "<feedback>", "</feedback>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    model.encoder.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    logger.info("Finish loading model [%s] from %s, type [%s]", get_model_size(model), args.model_name_or_path,args.model_type)

    if args.load_generator_path is not None:
        logger.info("Reload model from {}".format(args.load_generator_path))
        model.load_state_dict(torch.load(args.load_generator_path))

    return config, model, tokenizer


def eval_ppl_epoch(args, eval_dataloader, model, tokenizer,logger):
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_dataloader, eval_examples, model, tokenizer, split_tag, logger):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    p = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            preds = model.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    num_beams=args.beam_size,
                                    max_length=args.max_target_length,
                                    num_return_sequences=args.beam_size)
            preds = preds.reshape(source_ids.size(0), args.beam_size, -1)
            preds = preds.cpu().numpy()
            seq_preds = []
            for idx in range(preds.shape[0]):
                candidates = []
                for candidate in preds[idx]:
                    candidates.append(tokenizer.decode(candidate, skip_special_tokens=True,clean_up_tokenization_spaces=False))
                seq_preds.append(candidates)
            p.extend(seq_preds)

    output_dict = {}
    for idx,(ref,gold) in enumerate(zip(p,eval_examples)):
        output_dict[str(idx)] = (ref, gold["target"])
    with open(os.path.join(args.output_dir, f"{split_tag}_pred_gold.json"), 'w') as f:
        json.dump(output_dict, f, indent=2)

    em_1 = 0
    for key, sample in output_dict.items():
        predictions, ground_truth = sample[0], sample[1]
        prediction = predictions[0]
        if prediction.split('\t')[-1].strip() == ground_truth.split('\t')[-1].strip():
            em_1 += 1

    (goldMap, predictionMap) = bleu.computeMaps_multiple(os.path.join(args.output_dir, f"{split_tag}_pred_gold.json"), 1) 
    dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)

    result = {'em': round(em_1/len(output_dict)*100, 2), 'bleu': dev_bleu}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def train_generator(args: argparse.Namespace, epoch: int, generator: T5ForConditionalGeneration,
                    train_dataloader: DataLoader,eval_dataloader: DataLoader,
                    recording_variables: list, 
                    tokenizer: RobertaTokenizer,optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler.LambdaLR,
                    tb_writer: SummaryWriter,summary_file,logger: logging.Logger):
    """
    Func:
        Train 1 epoch of generator.
    """
    # record variables
    global_step, best_bleu_em, best_ppl, not_loss_dec_cnt, not_bleu_em_inc_cnt = recording_variables
    nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0

    generator.train()
    
    """
    Training of 1 epoch
    """
    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")

    for batch in bar:
        batch = tuple(t.to(args.device) for t in batch)

        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        outputs = generator(input_ids=source_ids, attention_mask=source_mask,
                        labels=target_ids, decoder_attention_mask=target_mask)
        loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()

        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if nb_tr_steps % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("[{}] Train loss {}".format(epoch, round(train_loss, 3)))

    # save last checkpoint
    if args.save_last_checkpoints:
        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = generator.module if hasattr(generator, 'module') else generator
        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Save the last model into %s", output_model_file)

    # save every epoch checkpoint
    every_epoch_output_dir = os.path.join(args.output_dir, 'checkpoint-all'.format(epoch))
    if not os.path.exists(every_epoch_output_dir):
        os.makedirs(every_epoch_output_dir)
    model_to_save = generator.module if hasattr(generator, 'module') else generator
    output_model_file = os.path.join(every_epoch_output_dir, f"pytorch_model_{epoch}.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Save the model of epoch %d into %s", epoch, output_model_file)
    """
    Eval of 1 epoch (ppl、blue)
    """
    if args.do_eval:
        """ eval ppl """
        eval_ppl = eval_ppl_epoch(args, eval_dataloader, generator, tokenizer,logger)

        # print the eval result
        result = {'epoch': epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  " + "*" * 20)

        if args.data_num == -1:
            tb_writer.add_scalar('dev_ppl', eval_ppl, epoch)

        # Save best checkpoint for best ppl
        early_stop = False
        if eval_ppl < best_ppl:
            not_loss_dec_cnt = 0
            logger.info("  Best ppl:%s", eval_ppl)
            logger.info("  " + "*" * 20)
            summary_file.write("[%d] Best ppl changed into %.4f\n" % (epoch, eval_ppl))
            best_ppl = eval_ppl

            output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if args.always_save_model:
                model_to_save = generator.module if hasattr(generator, 'module') else generator
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("Save the best ppl model into %s", output_model_file)
        else: # when ppl does not decrease in current epoch
            not_loss_dec_cnt += 1
            logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
            if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                    epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                logger.info(early_stop_str)
                summary_file.write(early_stop_str)
                early_stop = True

        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()

        """ eval bleu """
        if args.do_eval_bleu:
            eval_examples, eval_data = load_data(args, args.dev_filename, tokenizer, 'dev', only_src=True)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                        num_workers=4, pin_memory=True)

            result = eval_bleu_epoch(args, eval_dataloader, eval_examples, generator, tokenizer, 'dev', logger)
            dev_bleu, dev_em = result['bleu'], result['em']
            dev_bleu_em = dev_bleu + dev_em
            
            if args.data_num == -1:
                tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, epoch)

            # Save best checkpoint for best bleu
            if dev_bleu_em > best_bleu_em:
                not_bleu_em_inc_cnt = 0
                logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",epoch, dev_bleu_em, dev_bleu, dev_em)
                logger.info("  " + "*" * 20)
                best_bleu_em = dev_bleu_em
                summary_file.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                    epoch, best_bleu_em, dev_bleu, dev_em))
                
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args.data_num == -1 or args.always_save_model:
                    model_to_save = generator.module if hasattr(generator, 'module') else generator
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best bleu model into %s", output_model_file)
            else: # when dev_bleu does not increase in current epoch
                not_bleu_em_inc_cnt += 1
                logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                summary_file.write(
                    "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                        epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                    stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                        epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                    logger.info(stop_early_str)
                    summary_file.write(stop_early_str)
                    early_stop = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    return [global_step, best_bleu_em, best_ppl, not_loss_dec_cnt, not_bleu_em_inc_cnt], early_stop



def evaluate_generator(args: argparse.Namespace,model,
                    eval_dataloader: DataLoader, eval_examples: list, 
                    tokenizer: RobertaTokenizer,
                    summary_file,logger: logging.Logger):
    logger.info("  " + "***** Testing *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    results = []

    for criteria in ['last']:
        file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
        logger.info("Reload model from {}".format(file))
        if isinstance(model, DataParallel):
            model = model.module
        model.load_state_dict(torch.load(file))

        result = eval_bleu_epoch(args, eval_dataloader, eval_examples, model, tokenizer, 'test',logger)
        test_bleu, test_em = result['bleu'], result['em']
        test_codebleu = result['codebleu'] if 'codebleu' in result else 0

        result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
        logger.info(result_str)
        summary_file.write(result_str)
        
        results.append([test_bleu,test_em,test_codebleu,file,result_str])

    return results