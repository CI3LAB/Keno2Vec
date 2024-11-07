import logging
import torch
from torch.utils.data import TensorDataset, Dataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample_subsumption(object):
    def __init__(self, guid, entity_1, entity_2, entity_1_label, entity_2_label, label, entity_1_annotation, entity_2_annotation):
        self.guid = guid
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.entity_1_label = entity_1_label
        self.entity_2_label = entity_2_label
        self.label = label
        self.entity_1_annotation = entity_1_annotation
        self.entity_2_annotation = entity_2_annotation

class InputExample_property(object):
    def __init__(self, guid, entity_1, property, entity_2, entity_1_label, property_label, entity_2_label, label, entity_1_annotation, property_annotation, entity_2_annotation):
        self.guid = guid
        self.entity_1 = entity_1
        self.property = property
        self.entity_2 = entity_2
        self.entity_1_label = entity_1_label
        self.property_label = property_label
        self.entity_2_label = entity_2_label
        self.label = label
        self.entity_1_annotation = entity_1_annotation
        self.property_annotation = property_annotation
        self.entity_2_annotation = entity_2_annotation

class InputExample_multiple(object):
    def __init__(self, guid, entity, entity_label, label, entity_annotation):
        self.guid = guid
        self.entity = entity
        self.entity_label = entity_label
        self.label = label
        self.entity_annotation = entity_annotation

class InputExample_membership(object):
    def __init__(self, guid, instance, entity, instance_label, entity_label, label, instance_annotation, entity_annotation):
        self.guid = guid
        self.instance = instance
        self.entity = entity
        self.instance_label = instance_label
        self.entity_label = entity_label
        self.label = label
        self.instance_annotation = instance_annotation
        self.entity_annotation = entity_annotation

class InputFeature_subsumption(object):
    def __init__(self, guid, entity_1_input_ids, entity_1_attention_mask, entity_2_input_ids, entity_2_attention_mask,
                 entity_1_label_input_ids, entity_1_label_attention_mask, entity_2_label_input_ids, entity_2_label_attention_mask,
                 entity_1_annotation_input_ids, entity_1_annotation_attention_mask, entity_2_annotation_input_ids, entity_2_annotation_attention_mask, label_id):
        self.guid = guid
        self.entity_1_input_ids = entity_1_input_ids
        self.entity_1_attention_mask = entity_1_attention_mask
        self.entity_2_input_ids = entity_2_input_ids
        self.entity_2_attention_mask = entity_2_attention_mask
        self.entity_1_label_input_ids = entity_1_label_input_ids
        self.entity_1_label_attention_mask = entity_1_label_attention_mask
        self.entity_2_label_input_ids = entity_2_label_input_ids
        self.entity_2_label_attention_mask = entity_2_label_attention_mask
        self.entity_1_annotation_input_ids = entity_1_annotation_input_ids
        self.entity_1_annotation_attention_mask = entity_1_annotation_attention_mask
        self.entity_2_annotation_input_ids = entity_2_annotation_input_ids
        self.entity_2_annotation_attention_mask = entity_2_annotation_attention_mask
        self.label_id = label_id
    
    @staticmethod
    def collate_fct(batch):
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        max_len_1 = torch.sum(batch_tuple[1], dim=-1, keepdim=False).max().item()
        max_len_2 = torch.sum(batch_tuple[3], dim=-1, keepdim=False).max().item()
        max_len_3 = torch.sum(batch_tuple[5], dim=-1, keepdim=False).max().item()
        max_len_4 = torch.sum(batch_tuple[7], dim=-1, keepdim=False).max().item()
        max_len_5 = torch.sum(batch_tuple[9], dim=-1, keepdim=False).max().item()
        max_len_6 = torch.sum(batch_tuple[11], dim=-1, keepdim=False).max().item()
        results = ()
        for i, item in enumerate(batch_tuple):
            if item.dim() >= 2:
                if i < 2:
                    results += (item[:, :max_len_1],)
                elif i < 4:
                    results += (item[:, :max_len_2],)
                elif i < 6:
                    results += (item[:, :max_len_3],)
                elif i < 8:
                    results += (item[:, :max_len_4],)
                elif i < 10:
                    results += (item[:, :max_len_5],)
                else:
                    results += (item[:, :max_len_6],)
            else:
                results += (item,)
        return results

class InputFeature_property(object):
    def __init__(self, guid, entity_1_input_ids, entity_1_attention_mask, property_input_ids, property_attention_mask, entity_2_input_ids, entity_2_attention_mask,
                 entity_1_label_input_ids, entity_1_label_attention_mask, property_label_input_ids, property_label_attention_mask, entity_2_label_input_ids, entity_2_label_attention_mask,
                 entity_1_annotation_input_ids, entity_1_annotation_attention_mask, property_annotation_input_ids, property_annotation_attention_mask, entity_2_annotation_input_ids, entity_2_annotation_attention_mask, label_id):
        self.guid = guid
        self.entity_1_input_ids = entity_1_input_ids
        self.entity_1_attention_mask = entity_1_attention_mask
        self.property_input_ids = property_input_ids
        self.property_attention_mask = property_attention_mask
        self.entity_2_input_ids = entity_2_input_ids
        self.entity_2_attention_mask = entity_2_attention_mask
        self.entity_1_label_input_ids = entity_1_label_input_ids
        self.entity_1_label_attention_mask = entity_1_label_attention_mask
        self.property_label_input_ids = property_label_input_ids
        self.property_label_attention_mask = property_label_attention_mask
        self.entity_2_label_input_ids = entity_2_label_input_ids
        self.entity_2_label_attention_mask = entity_2_label_attention_mask
        self.entity_1_annotation_input_ids = entity_1_annotation_input_ids
        self.entity_1_annotation_attention_mask = entity_1_annotation_attention_mask
        self.property_annotation_input_ids = property_annotation_input_ids
        self.property_annotation_attention_mask = property_annotation_attention_mask
        self.entity_2_annotation_input_ids = entity_2_annotation_input_ids
        self.entity_2_annotation_attention_mask = entity_2_annotation_attention_mask
        self.label_id = label_id
    
    @staticmethod
    def collate_fct(batch):
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        max_len_1 = torch.sum(batch_tuple[1], dim=-1, keepdim=False).max().item()
        max_len_2 = torch.sum(batch_tuple[3], dim=-1, keepdim=False).max().item()
        max_len_3 = torch.sum(batch_tuple[5], dim=-1, keepdim=False).max().item()
        max_len_4 = torch.sum(batch_tuple[7], dim=-1, keepdim=False).max().item()
        max_len_5 = torch.sum(batch_tuple[9], dim=-1, keepdim=False).max().item()
        max_len_6 = torch.sum(batch_tuple[11], dim=-1, keepdim=False).max().item()
        max_len_7 = torch.sum(batch_tuple[13], dim=-1, keepdim=False).max().item()
        max_len_8 = torch.sum(batch_tuple[15], dim=-1, keepdim=False).max().item()
        max_len_9 = torch.sum(batch_tuple[17], dim=-1, keepdim=False).max().item()
        results = ()
        for i, item in enumerate(batch_tuple):
            if item.dim() >= 2:
                if i < 2:
                    results += (item[:, :max_len_1],)
                elif i < 4:
                    results += (item[:, :max_len_2],)
                elif i < 6:
                    results += (item[:, :max_len_3],)
                elif i < 8:
                    results += (item[:, :max_len_4],)
                elif i < 10:
                    results += (item[:, :max_len_5],)
                elif i < 12:
                    results += (item[:, :max_len_6],)
                elif i < 14:
                    results += (item[:, :max_len_7],)
                elif i < 16:
                    results += (item[:, :max_len_8],)
                else:
                    results += (item[:, :max_len_9],)
            else:
                results += (item,)
        return results

class InputFeature_multiple(object):
    def __init__(self, guid, entity_input_ids, entity_attention_mask, label_input_ids, label_attention_mask, annotation_input_ids, annotation_attention_mask, label_id):
        self.guid = guid
        self.entity_input_ids = entity_input_ids
        self.entity_attention_mask = entity_attention_mask
        self.label_input_ids = label_input_ids
        self.label_attention_mask = label_attention_mask
        self.annotation_input_ids = annotation_input_ids
        self.annotation_attention_mask = annotation_attention_mask
        self.label_id = label_id
    
    @staticmethod
    def collate_fct(batch):
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        max_len_1 = torch.sum(batch_tuple[1], dim=-1, keepdim=False).max().item()
        max_len_2 = torch.sum(batch_tuple[3], dim=-1, keepdim=False).max().item()
        max_len_3 = torch.sum(batch_tuple[5], dim=-1, keepdim=False).max().item()
        results = ()
        for i, item in enumerate(batch_tuple):
            if item.dim() >= 2:
                if i < 2:
                    results += (item[:, :max_len_1],)
                elif i < 4:
                    results += (item[:, :max_len_2],)
                else:
                    results += (item[:, :max_len_3],)
            else:
                results += (item,)
        return results

class InputFeature_membership(object):
    def __init__(self, guid, instance_input_ids, instance_attention_mask, entity_input_ids, entity_attention_mask, instance_label_input_ids, instance_label_attention_mask, entity_label_input_ids, entity_label_attention_mask, 
                 instance_annotation_input_ids, instance_annotation_attention_mask, entity_annotation_input_ids, entity_annotation_attention_mask, label_id):
        self.guid = guid
        self.instance_input_ids = instance_input_ids
        self.instance_attention_mask = instance_attention_mask
        self.entity_input_ids = entity_input_ids
        self.entity_attention_mask = entity_attention_mask
        self.instance_label_input_ids = instance_label_input_ids
        self.instance_label_attention_mask = instance_label_attention_mask
        self.entity_label_input_ids = entity_label_input_ids
        self.entity_label_attention_mask = entity_label_attention_mask
        self.instance_annotation_input_ids = instance_annotation_input_ids
        self.instance_annotation_attention_mask = instance_annotation_attention_mask
        self.entity_annotation_input_ids = entity_annotation_input_ids
        self.entity_annotation_attention_mask = entity_annotation_attention_mask
        self.label_id = label_id
    
    @staticmethod
    def collate_fct(batch):
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        max_len_1 = torch.sum(batch_tuple[1], dim=-1, keepdim=False).max().item()
        max_len_2 = torch.sum(batch_tuple[3], dim=-1, keepdim=False).max().item()
        max_len_3 = torch.sum(batch_tuple[5], dim=-1, keepdim=False).max().item()
        max_len_4 = torch.sum(batch_tuple[7], dim=-1, keepdim=False).max().item()
        max_len_5 = torch.sum(batch_tuple[9], dim=-1, keepdim=False).max().item()
        max_len_6 = torch.sum(batch_tuple[11], dim=-1, keepdim=False).max().item()
        results = ()
        for i, item in enumerate(batch_tuple):
            if item.dim() >= 2:
                if i < 2:
                    results += (item[:, :max_len_1],)
                elif i < 4:
                    results += (item[:, :max_len_2],)
                elif i < 6:
                    results += (item[:, :max_len_3],)
                elif i < 8:
                    results += (item[:, :max_len_4],)
                elif i < 10:
                    results += (item[:, :max_len_5],)
                else:
                    results += (item[:, :max_len_6],)
            else:
                results += (item,)
        return results

def tokenizer_with_type(text, max_length, args, tokenizer=None):
    if "bert" in args.model_type:
        tokenized_inputs = tokenizer(text, max_length=max_length, padding = 'max_length',
                                    truncation=True, add_special_tokens=True)
    else: # word2vec
        word2id = tokenizer
        token_ids = []
        text = text.lower() # uncased
        words = text.split(' ')
        for word in words:
            if word in word2id:
                token_ids.append(word2id.index(word))
            else:
                token_ids.append(word2id.index('<UNK>'))
        # get label_ids of each sentence
        if len(token_ids) < max_length:
            input_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
            token_ids += ([0] * (max_length - len(token_ids))) # index of <PAD> is 0
        else:
            input_mask = [1] * max_length
            token_ids = token_ids[:max_length]

        assert len(token_ids) == max_length
        assert len(input_mask) == max_length
        tokenized_inputs = {"input_ids": token_ids, "attention_mask": input_mask}
    
    return tokenized_inputs


def convert_examples_to_features(
    examples,
    max_entity_length,
    max_annotation_length,
    tokenizer,
    task,
    args
):
    features = []
    for example in examples:
        guid = example.guid
        
        if task == "subsumption":
            tokenized_inputs_e1 = tokenizer_with_type(example.entity_1, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e2 = tokenizer_with_type(example.entity_2, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e1l = tokenizer_with_type(example.entity_1_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e2l = tokenizer_with_type(example.entity_2_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e1a = tokenizer_with_type(example.entity_1_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e2a = tokenizer_with_type(example.entity_2_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            label_id = int(example.label)
        
            features.append(
                InputFeature_subsumption(
                    guid=guid,
                    entity_1_input_ids=tokenized_inputs_e1['input_ids'],
                    entity_1_attention_mask=tokenized_inputs_e1['attention_mask'],
                    entity_2_input_ids=tokenized_inputs_e2['input_ids'],
                    entity_2_attention_mask=tokenized_inputs_e2['attention_mask'],
                    entity_1_label_input_ids=tokenized_inputs_e1l['input_ids'],
                    entity_1_label_attention_mask=tokenized_inputs_e1l['attention_mask'],
                    entity_2_label_input_ids=tokenized_inputs_e2l['input_ids'],
                    entity_2_label_attention_mask=tokenized_inputs_e2l['attention_mask'],
                    entity_1_annotation_input_ids=tokenized_inputs_e1a['input_ids'],
                    entity_1_annotation_attention_mask=tokenized_inputs_e1a['attention_mask'],
                    entity_2_annotation_input_ids=tokenized_inputs_e2a['input_ids'],
                    entity_2_annotation_attention_mask=tokenized_inputs_e2a['attention_mask'],
                    label_id=label_id
                )
            )
        
        elif task == "property":
            tokenized_inputs_e1 = tokenizer_with_type(example.entity_1, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_p = tokenizer_with_type(example.property, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e2 = tokenizer_with_type(example.entity_2, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e1l = tokenizer_with_type(example.entity_1_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_pl = tokenizer_with_type(example.property_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e2l = tokenizer_with_type(example.entity_2_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e1a = tokenizer_with_type(example.entity_1_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_pa = tokenizer_with_type(example.property_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e2a = tokenizer_with_type(example.entity_2_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            label_id = int(example.label)
        
            features.append(
                InputFeature_property(
                    guid=guid,
                    entity_1_input_ids=tokenized_inputs_e1['input_ids'],
                    entity_1_attention_mask=tokenized_inputs_e1['attention_mask'],
                    property_input_ids=tokenized_inputs_p['input_ids'],
                    property_attention_mask=tokenized_inputs_p['attention_mask'],
                    entity_2_input_ids=tokenized_inputs_e2['input_ids'],
                    entity_2_attention_mask=tokenized_inputs_e2['attention_mask'],
                    entity_1_label_input_ids=tokenized_inputs_e1l['input_ids'],
                    entity_1_label_attention_mask=tokenized_inputs_e1l['attention_mask'],
                    property_label_input_ids=tokenized_inputs_pl['input_ids'],
                    property_label_attention_mask=tokenized_inputs_pl['attention_mask'],
                    entity_2_label_input_ids=tokenized_inputs_e2l['input_ids'],
                    entity_2_label_attention_mask=tokenized_inputs_e2l['attention_mask'],
                    entity_1_annotation_input_ids=tokenized_inputs_e1a['input_ids'],
                    entity_1_annotation_attention_mask=tokenized_inputs_e1a['attention_mask'],
                    property_annotation_input_ids=tokenized_inputs_pa['input_ids'],
                    property_annotation_attention_mask=tokenized_inputs_pa['attention_mask'],
                    entity_2_annotation_input_ids=tokenized_inputs_e2a['input_ids'],
                    entity_2_annotation_attention_mask=tokenized_inputs_e2a['attention_mask'],
                    label_id=label_id
                )
            )
        
        elif task == "multiple":
            tokenized_inputs_e = tokenizer_with_type(example.entity, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_l = tokenizer_with_type(example.entity_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_a = tokenizer_with_type(example.entity_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            label_id = int(example.label)
        
            features.append(
                InputFeature_multiple(
                    guid=guid,
                    entity_input_ids=tokenized_inputs_e['input_ids'],
                    entity_attention_mask=tokenized_inputs_e['attention_mask'],
                    label_input_ids=tokenized_inputs_l['input_ids'],
                    label_attention_mask=tokenized_inputs_l['attention_mask'],
                    annotation_input_ids=tokenized_inputs_a['input_ids'],
                    annotation_attention_mask=tokenized_inputs_a['attention_mask'],
                    label_id=label_id
                )
            )

        elif task == "membership":
            tokenized_inputs_i = tokenizer_with_type(example.instance, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_e = tokenizer_with_type(example.entity, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_il = tokenizer_with_type(example.instance_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_el = tokenizer_with_type(example.entity_label, max_length=max_entity_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_ia = tokenizer_with_type(example.instance_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            tokenized_inputs_ea = tokenizer_with_type(example.entity_annotation, max_length=max_annotation_length, args=args, tokenizer=tokenizer)
            label_id = int(example.label)
        
            features.append(
                InputFeature_membership(
                    guid=guid,
                    instance_input_ids=tokenized_inputs_i['input_ids'],
                    instance_attention_mask=tokenized_inputs_i['attention_mask'],
                    entity_input_ids=tokenized_inputs_e['input_ids'],
                    entity_attention_mask=tokenized_inputs_e['attention_mask'],
                    instance_label_input_ids=tokenized_inputs_il['input_ids'],
                    instance_label_attention_mask=tokenized_inputs_il['attention_mask'],
                    entity_label_input_ids=tokenized_inputs_el['input_ids'],
                    entity_label_attention_mask=tokenized_inputs_el['attention_mask'],
                    instance_annotation_input_ids=tokenized_inputs_ia['input_ids'],
                    instance_annotation_attention_mask=tokenized_inputs_ia['attention_mask'],
                    entity_annotation_input_ids=tokenized_inputs_ea['input_ids'],
                    entity_annotation_attention_mask=tokenized_inputs_ea['attention_mask'],
                    label_id=label_id
                )
            )
        
        else:
            raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")

    return features

def load_examples(args, data_processor, split, tokenizer=None):
    logger.info("Loading and converting data from data_utils.py...")
    # Load data features from dataset file
    if args.sample_ratio > 0 and split == "train":
        examples = data_processor.get_examples_sample(sample_ratio=args.sample_ratio, seed=args.seed, split=split)
    else:
        examples = data_processor.get_examples(split)

    features = convert_examples_to_features(
        examples,
        args.max_entity_length,
        args.max_annotation_length,
        tokenizer,
        task = args.task,
        args=args
    )

    if args.task == "subsumption":
        all_entity_1_input_ids = torch.tensor([f.entity_1_input_ids for f in features], dtype=torch.long)
        all_entity_1_attention_mask = torch.tensor([f.entity_1_attention_mask for f in features], dtype=torch.long)
        all_entity_2_input_ids = torch.tensor([f.entity_2_input_ids for f in features], dtype=torch.long)
        all_entity_2_attention_mask = torch.tensor([f.entity_2_attention_mask for f in features], dtype=torch.long)
        all_entity_1_label_input_ids = torch.tensor([f.entity_1_label_input_ids for f in features], dtype=torch.long)
        all_entity_1_label_attention_mask = torch.tensor([f.entity_1_label_attention_mask for f in features], dtype=torch.long)
        all_entity_2_label_input_ids = torch.tensor([f.entity_2_label_input_ids for f in features], dtype=torch.long)
        all_entity_2_label_attention_mask = torch.tensor([f.entity_2_label_attention_mask for f in features], dtype=torch.long)
        all_entity_1_annotation_input_ids = torch.tensor([f.entity_1_annotation_input_ids for f in features], dtype=torch.long)
        all_entity_1_annotation_attention_mask = torch.tensor([f.entity_1_annotation_attention_mask for f in features], dtype=torch.long)
        all_entity_2_annotation_input_ids = torch.tensor([f.entity_2_annotation_input_ids for f in features], dtype=torch.long)
        all_entity_2_annotation_attention_mask = torch.tensor([f.entity_2_annotation_attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_entity_1_input_ids, all_entity_1_attention_mask, all_entity_2_input_ids, all_entity_2_attention_mask,
                                all_entity_1_label_input_ids, all_entity_1_label_attention_mask, all_entity_2_label_input_ids, all_entity_2_label_attention_mask,
                                all_entity_1_annotation_input_ids, all_entity_1_annotation_attention_mask, all_entity_2_annotation_input_ids, all_entity_2_annotation_attention_mask, all_label_ids)
    
    elif args.task == "property":
        all_entity_1_input_ids = torch.tensor([f.entity_1_input_ids for f in features], dtype=torch.long)
        all_entity_1_attention_mask = torch.tensor([f.entity_1_attention_mask for f in features], dtype=torch.long)
        all_property_input_ids = torch.tensor([f.property_input_ids for f in features], dtype=torch.long)
        all_property_attention_mask = torch.tensor([f.property_attention_mask for f in features], dtype=torch.long)
        all_entity_2_input_ids = torch.tensor([f.entity_2_input_ids for f in features], dtype=torch.long)
        all_entity_2_attention_mask = torch.tensor([f.entity_2_attention_mask for f in features], dtype=torch.long)
        all_entity_1_label_input_ids = torch.tensor([f.entity_1_label_input_ids for f in features], dtype=torch.long)
        all_entity_1_label_attention_mask = torch.tensor([f.entity_1_label_attention_mask for f in features], dtype=torch.long)
        all_property_label_input_ids = torch.tensor([f.property_label_input_ids for f in features], dtype=torch.long)
        all_property_label_attention_mask = torch.tensor([f.property_label_attention_mask for f in features], dtype=torch.long)
        all_entity_2_label_input_ids = torch.tensor([f.entity_2_label_input_ids for f in features], dtype=torch.long)
        all_entity_2_label_attention_mask = torch.tensor([f.entity_2_label_attention_mask for f in features], dtype=torch.long)
        all_entity_1_annotation_input_ids = torch.tensor([f.entity_1_annotation_input_ids for f in features], dtype=torch.long)
        all_entity_1_annotation_attention_mask = torch.tensor([f.entity_1_annotation_attention_mask for f in features], dtype=torch.long)
        all_property_annotation_input_ids = torch.tensor([f.property_annotation_input_ids for f in features], dtype=torch.long)
        all_property_annotation_attention_mask = torch.tensor([f.property_annotation_attention_mask for f in features], dtype=torch.long)
        all_entity_2_annotation_input_ids = torch.tensor([f.entity_2_annotation_input_ids for f in features], dtype=torch.long)
        all_entity_2_annotation_attention_mask = torch.tensor([f.entity_2_annotation_attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_entity_1_input_ids, all_entity_1_attention_mask, all_property_input_ids, all_property_attention_mask, all_entity_2_input_ids, all_entity_2_attention_mask,
                                all_entity_1_label_input_ids, all_entity_1_label_attention_mask, all_property_label_input_ids, all_property_label_attention_mask, all_entity_2_label_input_ids, all_entity_2_label_attention_mask,
                                all_entity_1_annotation_input_ids, all_entity_1_annotation_attention_mask, all_property_annotation_input_ids, all_property_annotation_attention_mask, all_entity_2_annotation_input_ids, all_entity_2_annotation_attention_mask, all_label_ids)

    elif args.task == "multiple":
        all_entity_input_ids = torch.tensor([f.entity_input_ids for f in features], dtype=torch.long)
        all_entity_attention_mask = torch.tensor([f.entity_attention_mask for f in features], dtype=torch.long)
        all_label_input_ids = torch.tensor([f.label_input_ids for f in features], dtype=torch.long)
        all_label_attention_mask = torch.tensor([f.label_attention_mask for f in features], dtype=torch.long)
        all_annotation_input_ids = torch.tensor([f.annotation_input_ids for f in features], dtype=torch.long)
        all_annotation_attention_mask = torch.tensor([f.annotation_attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_entity_input_ids, all_entity_attention_mask, all_label_input_ids, all_label_attention_mask, all_annotation_input_ids, all_annotation_attention_mask, all_label_ids)
    
    elif args.task == "membership":
        all_instance_input_ids = torch.tensor([f.instance_input_ids for f in features], dtype=torch.long)
        all_instance_attention_mask = torch.tensor([f.instance_attention_mask for f in features], dtype=torch.long)
        all_entity_input_ids = torch.tensor([f.entity_input_ids for f in features], dtype=torch.long)
        all_entity_attention_mask = torch.tensor([f.entity_attention_mask for f in features], dtype=torch.long)
        all_instance_label_input_ids = torch.tensor([f.instance_label_input_ids for f in features], dtype=torch.long)
        all_instance_label_attention_mask = torch.tensor([f.instance_label_attention_mask for f in features], dtype=torch.long)
        all_entity_label_input_ids = torch.tensor([f.entity_label_input_ids for f in features], dtype=torch.long)
        all_entity_label_attention_mask = torch.tensor([f.entity_label_attention_mask for f in features], dtype=torch.long)
        all_instance_annotation_input_ids = torch.tensor([f.instance_annotation_input_ids for f in features], dtype=torch.long)
        all_instance_annotation_attention_mask = torch.tensor([f.instance_annotation_attention_mask for f in features], dtype=torch.long)
        all_entity_annotation_input_ids = torch.tensor([f.entity_annotation_input_ids for f in features], dtype=torch.long)
        all_entity_annotation_attention_mask = torch.tensor([f.entity_annotation_attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_instance_input_ids, all_instance_attention_mask, all_entity_input_ids, all_entity_attention_mask, all_instance_label_input_ids, all_instance_label_attention_mask,
                                all_entity_label_input_ids, all_entity_label_attention_mask, all_instance_annotation_input_ids, all_instance_annotation_attention_mask, all_entity_annotation_input_ids, all_entity_annotation_attention_mask, all_label_ids)
    
    else:
        raise ValueError("Invalid task. Choose between 'subsumption', 'property', 'multiple', and 'membership'.")
        
    return dataset