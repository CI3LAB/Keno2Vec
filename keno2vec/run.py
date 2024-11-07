from arguments import get_args_parser
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
from utils.data_processor import DataProcessor_subsumption, DataProcessor_property, DataProcessor_multiple, DataProcessor_membership
from utils.data_utils import InputExample_subsumption, InputExample_property, InputExample_multiple, InputExample_membership, \
                             InputFeature_subsumption, InputFeature_property, InputFeature_multiple, InputFeature_membership, \
                             load_examples
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from models.models import Model4Subsumption, Model4Property, Model4Multiple, Model4Membership
# from torch.utils.tensorboard import SummaryWriter
import json
import re
from safetensors import safe_open
from owlready2 import *

logger = logging.getLogger(__name__)
# writer = SummaryWriter(log_dir=f"./runs")

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_scores_binary(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def get_scores_multiclass(y_pred, y_true):
    micro_f1 = round(f1_score(y_true, y_pred, average='micro') * 100, 2)
    macro_f1 = round(f1_score(y_true, y_pred, average='macro') * 100, 2)
    weighted_f1 = round(f1_score(y_true, y_pred, average='weighted') * 100, 2)
    return macro_f1, weighted_f1, micro_f1
    

def train(args, train_dataset, dev_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    if args.task == "subsumption":
        collate_fn = InputFeature_subsumption.collate_fct
    elif args.task == "property":
        collate_fn = InputFeature_property.collate_fct
    elif args.task == "multiple":
        collate_fn = InputFeature_multiple.collate_fct
    elif args.task == "membership":
        collate_fn = InputFeature_membership.collate_fct
    else:
        raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")
    
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn
                                  )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps
    
    # Prepare optimizer and schedule (linear warmup and decay), freeze should add 'and "bert" not in n'
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        "lr": args.learning_rate},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": args.learning_rate},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_proportion != 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Random seed = %d", args.seed)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    steps_trained_in_current_epoch = 0
    logging_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproductibility
    epoch_num = 1
    # torch.autograd.set_detect_anomaly(True)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.task == "subsumption":
                inputs = {
                    "entity_1_input_ids":batch[0],
                    "entity_1_attention_mask":batch[1],
                    "entity_2_input_ids":batch[2],
                    "entity_2_attention_mask":batch[3],
                    "entity_1_label_input_ids":batch[4],
                    "entity_1_label_attention_mask":batch[5],
                    "entity_2_label_input_ids":batch[6],
                    "entity_2_label_attention_mask":batch[7],
                    "entity_1_annotation_input_ids":batch[8],
                    "entity_1_annotation_attention_mask":batch[9],
                    "entity_2_annotation_input_ids":batch[10],
                    "entity_2_annotation_attention_mask":batch[11],
                    "label_ids":batch[12],
                    "mode":"train"
                }
            elif args.task == "property":
                inputs = {
                    "entity_1_input_ids":batch[0],
                    "entity_1_attention_mask":batch[1],
                    "property_input_ids":batch[2],
                    "property_attention_mask":batch[3],
                    "entity_2_input_ids":batch[4],
                    "entity_2_attention_mask":batch[5],
                    "entity_1_label_input_ids":batch[6],
                    "entity_1_label_attention_mask":batch[7],
                    "property_label_input_ids":batch[8],
                    "property_label_attention_mask":batch[9],
                    "entity_2_label_input_ids":batch[10],
                    "entity_2_label_attention_mask":batch[11],
                    "entity_1_annotation_input_ids":batch[12],
                    "entity_1_annotation_attention_mask":batch[13],
                    "property_annotation_input_ids":batch[14],
                    "property_annotation_attention_mask":batch[15],
                    "entity_2_annotation_input_ids":batch[16],
                    "entity_2_annotation_attention_mask":batch[17],
                    "label_ids":batch[18],
                    "mode":"train"
                }
            elif args.task == "multiple":
                inputs = {
                    "entity_input_ids":batch[0],
                    "entity_attention_mask":batch[1],
                    "label_input_ids":batch[2],
                    "label_attention_mask":batch[3],
                    "annotation_input_ids":batch[4],
                    "annotation_attention_mask":batch[5],
                    "label_ids":batch[6],
                    "mode":"train"
                }
            elif args.task == "membership":
                inputs = {
                    "instance_input_ids":batch[0],
                    "instance_attention_mask":batch[1],
                    "entity_input_ids":batch[2],
                    "entity_attention_mask":batch[3],
                    "instance_label_input_ids":batch[4],
                    "instance_label_attention_mask":batch[5],
                    "entity_label_input_ids":batch[6],
                    "entity_label_attention_mask":batch[7],
                    "instance_annotation_input_ids":batch[8],
                    "instance_annotation_attention_mask":batch[9],
                    "entity_annotation_input_ids":batch[10],
                    "entity_annotation_attention_mask":batch[11],
                    "label_ids":batch[12],
                    "mode":"train"
                }
            else:
                raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")
            
            loss = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            tr_loss += loss.item()
            logging_loss += loss.item()

            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # writer.add_scalar("loss", loss.item(), global_step)
                # writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
        # evaluate after every epoch
        if args.evaluate_after_epoch:
            logging_loss = logging_loss / (step + 1)
            print('Training Loss: {}'.format(logging_loss))

            s1, s2, f1, f1_str = evaluate(args, dev_dataset, model, tokenizer)
            print(f1_str)
            #output_eval_file = os.path.join(args.output_dir, "train_results.txt")
            #with open(output_eval_file, "a") as f:
            #    f.write('***** Predict Result for Dataset {} Seed {} *****\n'.format(args.data_dir, args.seed))
            #    f.write(result_str)
            #writer.add_scalar("eval_loss", eval_loss, epoch_num)
            #writer.add_scalar("trigger_f1", trigger_f1, epoch_num)

            if best_score <= f1:
                best_score = f1
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                # tokenizer.save_pretrained(output_dir)

                logger.info("Saving model checkpoint to %s", output_dir)
            
        epoch_num += 1

    return global_step, tr_loss / global_step

def evaluate(args, dev_dataset, model, tokenizer):
    # Eval!
    dev_sampler = SequentialSampler(dev_dataset)
    if args.task == "subsumption":
        collate_fn = InputFeature_subsumption.collate_fct
    elif args.task == "property":
        collate_fn = InputFeature_property.collate_fct
    elif args.task == "multiple":
        collate_fn = InputFeature_multiple.collate_fct
    elif args.task == "membership":
        collate_fn = InputFeature_membership.collate_fct
    else:
        raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")
    dev_dataloader = DataLoader(dev_dataset,
                                 sampler=dev_sampler,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=collate_fn
                                 )
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()

    preds = []
    trues = []
    for batch in tqdm(dev_dataloader, desc='Evaluating'):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.task == "subsumption":
                inputs = {
                    "entity_1_input_ids":batch[0],
                    "entity_1_attention_mask":batch[1],
                    "entity_2_input_ids":batch[2],
                    "entity_2_attention_mask":batch[3],
                    "entity_1_label_input_ids":batch[4],
                    "entity_1_label_attention_mask":batch[5],
                    "entity_2_label_input_ids":batch[6],
                    "entity_2_label_attention_mask":batch[7],
                    "entity_1_annotation_input_ids":batch[8],
                    "entity_1_annotation_attention_mask":batch[9],
                    "entity_2_annotation_input_ids":batch[10],
                    "entity_2_annotation_attention_mask":batch[11],
                    "label_ids":batch[12],
                    "mode":"test"
                }
            elif args.task == "property":
                inputs = {
                    "entity_1_input_ids":batch[0],
                    "entity_1_attention_mask":batch[1],
                    "property_input_ids":batch[2],
                    "property_attention_mask":batch[3],
                    "entity_2_input_ids":batch[4],
                    "entity_2_attention_mask":batch[5],
                    "entity_1_label_input_ids":batch[6],
                    "entity_1_label_attention_mask":batch[7],
                    "property_label_input_ids":batch[8],
                    "property_label_attention_mask":batch[9],
                    "entity_2_label_input_ids":batch[10],
                    "entity_2_label_attention_mask":batch[11],
                    "entity_1_annotation_input_ids":batch[12],
                    "entity_1_annotation_attention_mask":batch[13],
                    "property_annotation_input_ids":batch[14],
                    "property_annotation_attention_mask":batch[15],
                    "entity_2_annotation_input_ids":batch[16],
                    "entity_2_annotation_attention_mask":batch[17],
                    "label_ids":batch[18],
                    "mode":"test"
                }
            elif args.task == "multiple":
                inputs = {
                    "entity_input_ids":batch[0],
                    "entity_attention_mask":batch[1],
                    "label_input_ids":batch[2],
                    "label_attention_mask":batch[3],
                    "annotation_input_ids":batch[4],
                    "annotation_attention_mask":batch[5],
                    "label_ids":batch[6],
                    "mode":"test"
                }
            elif args.task == "membership":
                inputs = {
                    "instance_input_ids":batch[0],
                    "instance_attention_mask":batch[1],
                    "entity_input_ids":batch[2],
                    "entity_attention_mask":batch[3],
                    "instance_label_input_ids":batch[4],
                    "instance_label_attention_mask":batch[5],
                    "entity_label_input_ids":batch[6],
                    "entity_label_attention_mask":batch[7],
                    "instance_annotation_input_ids":batch[8],
                    "instance_annotation_attention_mask":batch[9],
                    "entity_annotation_input_ids":batch[10],
                    "entity_annotation_attention_mask":batch[11],
                    "label_ids":batch[12],
                    "mode":"test"
                }
            else:
                raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")
            

            loss, logits, label_ids = model(**inputs)
            assert logits.size() == label_ids.size()
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            eval_loss += loss.item()
            nb_eval_steps += 1

            for i in range(len(logits)):
                preds.append(logits[i])
                trues.append(label_ids[i])

    assert len(preds) == len(trues)

    # preds_down = [int(pred) for pred in preds]
    # with open('preds.json', 'w', encoding="utf-8") as f:
    #     json.dump(preds_down, f, ensure_ascii=False)

    eval_loss /= nb_eval_steps

    print('eval_loss={}'.format(eval_loss))

    if args.task != "multiple":
        precision, recall, f1 = get_scores_binary(preds, trues)
        f1_str = '[Precision]\t{:.4f}\n'.format(precision)
        f1_str += '[Recall]\t{:.4f}\n'.format(recall)
        f1_str += '[F1 Score]\t{:.4f}\n'.format(f1)
        return precision, recall, f1, f1_str
    else:
        macro_f1, weighted_f1, micro_f1 = get_scores_multiclass(preds, trues)
        f1_str = '[Micro F1]\t{:.4f}\n'.format(micro_f1)
        f1_str += '[Macro F1]\t{:.4f}\n'.format(macro_f1)
        f1_str += '[Weighted F1]\t{:.4f}\n'.format(weighted_f1)
        return macro_f1, weighted_f1, micro_f1, f1_str
    
def main():
    args = get_args_parser()

    args.device = torch.device("cuda")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    set_seed(args)

    logger.info("Loading dataset from run.py...")

    args.ontology = get_ontology(args.owl_path).load()

    if args.task == "subsumption":
        data_processor = DataProcessor_subsumption(args)
        model = Model4Subsumption(args, data_processor)
    elif args.task == "property":
        data_processor = DataProcessor_property(args)
        model = Model4Property(args, data_processor)
    elif args.task == "multiple":
        data_processor = DataProcessor_multiple(args)
        args.num_labels = len(data_processor.all_labels)
        model = Model4Multiple(args, data_processor)
    elif args.task == "membership":
        data_processor = DataProcessor_membership(args)
        model = Model4Membership(args, data_processor)
    else:
        raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")

    logger.info("Task: {}".format(args.task))
    
    if "bert" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = data_processor.word2id

    model = model.cuda()
    # for param in model.named_parameters():
    #     if 'label_embedding' in param[0]:
    #         print(param)
    # exit()
    
    # Training
    if args.do_train:
        train_dataset = load_examples(args, data_processor, 'train', tokenizer)
        dev_dataset = load_examples(args, data_processor, 'dev', tokenizer)

        global_step, tr_loss = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        output_dir = os.path.join(args.output_dir, "best_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        logger.info("Saving model checkpoint to %s", output_dir)
    
    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)

        model.to(args.device)

        dev_dataset = load_examples(args, data_processor, 'test', tokenizer)
        s1, s2, f1, f1_str = evaluate(args, dev_dataset, model, tokenizer)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        print(f1_str)
        with open(output_eval_file, "a") as f:
            f.write('***** Predict Result for Dataset {} Seed {} *****\n'.format(args.data_dir, args.seed))
            f.write(f1_str)
            f.write('\n')
    

if __name__ == "__main__":
    main()
