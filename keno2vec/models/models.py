from logging import log
import torch
import torch.nn as nn
from arguments import get_model_classes, get_args
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import re
import os
from safetensors import safe_open

class Model4Subsumption(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()
        if "bert" in args.model_type:
            model_classes = get_model_classes()
            model_config = model_classes[args.model_type]
            self.bert = model_config['model'].from_pretrained(
                args.model_name_or_path
            )
            self.hidden_size = self.bert.config.hidden_size
        else:
            self.hidden_size = args.embedding_dim
            self.bert = nn.Embedding(len(data_processor.word2id), self.hidden_size)
            self.bert.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
            self.bert.weight.requires_grad = True
        self.model_type = args.model_type
        self.annotation_free = args.annotation_free
        self.name_label = args.name_label
        self.W = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if args.continue_training:
            with safe_open(args.pretrained_model_path, framework="pt", device="cpu") as f:
                state_dict = {key: torch.tensor(f.get_tensor(key)) for key in f.keys()}
            bert_state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('bert')}
            self.bert.load_state_dict(bert_state_dict, strict=False)
        self.num_labels = 2
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_labels)
    
    def get_embedding(self, input_ids, attention_mask, model_type):
        if "bert" in model_type:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0]
        else:
            embedded = self.bert(input_ids)
            mask = attention_mask.unsqueeze(-1).expand(embedded.size()).float()
            masked_embed = embedded * mask
            sum_embeddings = torch.sum(masked_embed, dim=1)  # (bs, embedding_dim)
            num_non_pad = torch.sum(attention_mask, dim=1).unsqueeze(-1)  # (bs, 1)
            average_embed = sum_embeddings / num_non_pad  # (bs, embedding_dim)       
            return average_embed

    def forward(self, entity_1_input_ids, entity_1_attention_mask, entity_2_input_ids, entity_2_attention_mask,
                entity_1_label_input_ids, entity_1_label_attention_mask, entity_2_label_input_ids, entity_2_label_attention_mask,
                entity_1_annotation_input_ids, entity_1_annotation_attention_mask, entity_2_annotation_input_ids, entity_2_annotation_attention_mask, label_ids, mode):
        if self.name_label == "name":
            entity_1_outputs = self.get_embedding(input_ids=entity_1_input_ids, attention_mask=entity_1_attention_mask, model_type=self.model_type)
            entity_2_outputs = self.get_embedding(input_ids=entity_2_input_ids, attention_mask=entity_2_attention_mask, model_type=self.model_type)
        elif self.name_label == "label":
            entity_1_outputs = self.get_embedding(input_ids=entity_1_label_input_ids, attention_mask=entity_1_label_attention_mask, model_type=self.model_type)
            entity_2_outputs = self.get_embedding(input_ids=entity_2_label_input_ids, attention_mask=entity_2_label_attention_mask, model_type=self.model_type)
        else:
            entity_1_outputs = self.get_embedding(input_ids=entity_1_input_ids, attention_mask=entity_1_attention_mask, model_type=self.model_type)
            entity_2_outputs = self.get_embedding(input_ids=entity_2_input_ids, attention_mask=entity_2_attention_mask, model_type=self.model_type)
            entity_1_label_outputs = self.get_embedding(input_ids=entity_1_label_input_ids, attention_mask=entity_1_label_attention_mask, model_type=self.model_type)
            entity_2_label_outputs = self.get_embedding(input_ids=entity_2_label_input_ids, attention_mask=entity_2_label_attention_mask, model_type=self.model_type)
            entity_1_outputs = entity_1_outputs + entity_1_label_outputs
            entity_2_outputs = entity_2_outputs + entity_2_label_outputs
        
        if not self.annotation_free:
            entity_1_annotation_outputs = self.get_embedding(input_ids=entity_1_annotation_input_ids, attention_mask=entity_1_annotation_attention_mask, model_type=self.model_type)
            entity_2_annotation_outputs = self.get_embedding(input_ids=entity_2_annotation_input_ids, attention_mask=entity_2_annotation_attention_mask, model_type=self.model_type)

            entity_1_combined = torch.cat((entity_1_outputs, entity_1_annotation_outputs), dim=1) # bs, 2*hidden_size
            entity_2_combined = torch.cat((entity_2_outputs, entity_2_annotation_outputs), dim=1)

            entity_1_weight = self.sigmoid(self.W(entity_1_combined)) # bs, hidden_size
            entity_2_weight = self.sigmoid(self.W(entity_2_combined))

            entity_1_fused = entity_1_weight * entity_1_outputs + (1 - entity_1_weight) * entity_1_annotation_outputs # bs, hidden_size
            entity_2_fused = entity_2_weight * entity_2_outputs + (1 - entity_2_weight) * entity_2_annotation_outputs

            entity_1_merged = entity_1_outputs + entity_1_fused
            entity_2_merged = entity_2_outputs + entity_2_fused
        else:
            entity_1_merged = entity_1_outputs
            entity_2_merged = entity_2_outputs

        sequence_output = torch.cat((entity_1_merged, entity_2_merged), dim=1) # bs, 2*hidden_size
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else:
            logits = logits.argmax(-1)
            return loss, logits, label_ids
        
class Model4Property(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()
        if "bert" in args.model_type:
            model_classes = get_model_classes()
            model_config = model_classes[args.model_type]
            self.bert = model_config['model'].from_pretrained(
                args.model_name_or_path
            )
            self.hidden_size = self.bert.config.hidden_size
        else:
            self.hidden_size = args.embedding_dim
            self.bert = nn.Embedding(len(data_processor.word2id), self.hidden_size)
            self.bert.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
            self.bert.weight.requires_grad = True
        self.model_type = args.model_type
        self.annotation_free = args.annotation_free
        self.name_label = args.name_label
        self.W = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if args.continue_training:
            with safe_open(args.pretrained_model_path, framework="pt", device="cpu") as f:
                state_dict = {key: torch.tensor(f.get_tensor(key)) for key in f.keys()}
            bert_state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('bert')}
            self.bert.load_state_dict(bert_state_dict, strict=False)
        self.num_labels = 2
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size * 3, self.num_labels)

    def get_embedding(self, input_ids, attention_mask, model_type):
        if "bert" in model_type:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0]
        else:
            embedded = self.bert(input_ids)
            mask = attention_mask.unsqueeze(-1).expand(embedded.size()).float()
            masked_embed = embedded * mask
            sum_embeddings = torch.sum(masked_embed, dim=1)  # (bs, embedding_dim)
            num_non_pad = torch.sum(attention_mask, dim=1).unsqueeze(-1)  # (bs, 1)
            average_embed = sum_embeddings / num_non_pad  # (bs, embedding_dim)       
            return average_embed

    def forward(self, entity_1_input_ids, entity_1_attention_mask, property_input_ids, property_attention_mask, entity_2_input_ids, entity_2_attention_mask,
                 entity_1_label_input_ids, entity_1_label_attention_mask, property_label_input_ids, property_label_attention_mask, entity_2_label_input_ids, entity_2_label_attention_mask,
                 entity_1_annotation_input_ids, entity_1_annotation_attention_mask, property_annotation_input_ids, property_annotation_attention_mask, entity_2_annotation_input_ids, entity_2_annotation_attention_mask, label_ids, mode):
        if self.name_label == "name":
            entity_1_outputs = self.get_embedding(input_ids=entity_1_input_ids, attention_mask=entity_1_attention_mask, model_type=self.model_type)
            property_outputs = self.get_embedding(input_ids=property_input_ids, attention_mask=property_attention_mask, model_type=self.model_type)
            entity_2_outputs = self.get_embedding(input_ids=entity_2_input_ids, attention_mask=entity_2_attention_mask, model_type=self.model_type)
        elif self.name_label == "label":
            entity_1_outputs = self.get_embedding(input_ids=entity_1_label_input_ids, attention_mask=entity_1_label_attention_mask, model_type=self.model_type)
            property_outputs = self.get_embedding(input_ids=property_label_input_ids, attention_mask=property_label_attention_mask, model_type=self.model_type)
            entity_2_outputs = self.get_embedding(input_ids=entity_2_label_input_ids, attention_mask=entity_2_label_attention_mask, model_type=self.model_type)
        else:
            entity_1_outputs = self.get_embedding(input_ids=entity_1_input_ids, attention_mask=entity_1_attention_mask, model_type=self.model_type)
            property_outputs = self.get_embedding(input_ids=property_input_ids, attention_mask=property_attention_mask, model_type=self.model_type)
            entity_2_outputs = self.get_embedding(input_ids=entity_2_input_ids, attention_mask=entity_2_attention_mask, model_type=self.model_type)
            entity_1_label_outputs = self.get_embedding(input_ids=entity_1_label_input_ids, attention_mask=entity_1_label_attention_mask, model_type=self.model_type)
            property_label_outputs = self.get_embedding(input_ids=property_label_input_ids, attention_mask=property_label_attention_mask, model_type=self.model_type)
            entity_2_label_outputs = self.get_embedding(input_ids=entity_2_label_input_ids, attention_mask=entity_2_label_attention_mask, model_type=self.model_type)
            entity_1_outputs = entity_1_outputs + entity_1_label_outputs
            property_outputs = property_outputs + property_label_outputs
            entity_2_outputs = entity_2_outputs + entity_2_label_outputs
        
        if not self.annotation_free:
            entity_1_annotation_outputs = self.get_embedding(input_ids=entity_1_annotation_input_ids, attention_mask=entity_1_annotation_attention_mask, model_type=self.model_type)
            property_annotation_outputs = self.get_embedding(input_ids=property_annotation_input_ids, attention_mask=property_annotation_attention_mask, model_type=self.model_type)
            entity_2_annotation_outputs = self.get_embedding(input_ids=entity_2_annotation_input_ids, attention_mask=entity_2_annotation_attention_mask, model_type=self.model_type)

            combined_1 = torch.cat((entity_1_outputs, entity_1_annotation_outputs), dim=1) # bs, 2*hidden_size
            combined_p = torch.cat((property_outputs, property_annotation_outputs), dim=1)
            combined_2 = torch.cat((entity_2_outputs, entity_2_annotation_outputs), dim=1)

            weight_1 = self.sigmoid(self.W(combined_1)) # bs, hidden_size
            weight_p = self.sigmoid(self.W(combined_p))
            weight_2 = self.sigmoid(self.W(combined_2))

            entity_1_fused = weight_1 * entity_1_outputs + (1 - weight_1) * entity_1_annotation_outputs # bs, hidden_size
            property_fused = weight_p * property_outputs + (1 - weight_p) * property_annotation_outputs
            entity_2_fused = weight_2 * entity_2_outputs + (1 - weight_2) * entity_2_annotation_outputs

            entity_merged_1 = entity_1_outputs + entity_1_fused
            property_merged = property_outputs + property_fused
            entity_merged_2 = entity_2_outputs + entity_2_fused
        else:
            entity_merged_1 = entity_1_outputs
            property_merged = property_outputs
            entity_merged_2 = entity_2_outputs

        sequence_output = torch.cat((entity_merged_1, property_merged, entity_merged_2), dim=1) # bs, 3*hidden_size
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else:
            logits = logits.argmax(-1)
            return loss, logits, label_ids

class Model4Multiple(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()
        if "bert" in args.model_type:
            model_classes = get_model_classes()
            model_config = model_classes[args.model_type]
            self.bert = model_config['model'].from_pretrained(
                args.model_name_or_path
            )
            self.hidden_size = self.bert.config.hidden_size
        else:
            self.hidden_size = args.embedding_dim
            self.bert = nn.Embedding(len(data_processor.word2id), self.hidden_size)
            self.bert.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
            self.bert.weight.requires_grad = True
        self.model_type = args.model_type
        self.annotation_free = args.annotation_free
        self.name_label = args.name_label
        self.W = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if args.continue_training:
            with safe_open(args.pretrained_model_path, framework="pt", device="cpu") as f:
                state_dict = {key: torch.tensor(f.get_tensor(key)) for key in f.keys()}
            bert_state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('bert')}
            self.bert.load_state_dict(bert_state_dict, strict=False)
        self.num_labels = args.num_labels
        self.sigmoid = nn.Sigmoid() 
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
    
    def get_embedding(self, input_ids, attention_mask, model_type):
        if "bert" in model_type:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0]
        else:
            embedded = self.bert(input_ids)
            mask = attention_mask.unsqueeze(-1).expand(embedded.size()).float()
            masked_embed = embedded * mask
            sum_embeddings = torch.sum(masked_embed, dim=1)  # (bs, embedding_dim)
            num_non_pad = torch.sum(attention_mask, dim=1).unsqueeze(-1)  # (bs, 1)
            average_embed = sum_embeddings / num_non_pad  # (bs, embedding_dim)       
            return average_embed

    def forward(self, entity_input_ids, entity_attention_mask, label_input_ids, label_attention_mask, annotation_input_ids, annotation_attention_mask, label_ids, mode):
        if self.name_label == "name":
            entity_outputs = self.get_embedding(input_ids=entity_input_ids, attention_mask=entity_attention_mask, model_type=self.model_type)
        elif self.name_label == "label":
            entity_outputs = self.get_embedding(input_ids=label_input_ids, attention_mask=label_attention_mask, model_type=self.model_type)
        else:
            entity_outputs = self.get_embedding(input_ids=entity_input_ids, attention_mask=entity_attention_mask, model_type=self.model_type)
            label_outputs = self.get_embedding(input_ids=label_input_ids, attention_mask=label_attention_mask, model_type=self.model_type)
            entity_outputs = entity_outputs + label_outputs
        
        if not self.annotation_free:
            entity_annotation_outputs = self.get_embedding(input_ids=annotation_input_ids, attention_mask=annotation_attention_mask, model_type=self.model_type)

            combined = torch.cat((entity_outputs, entity_annotation_outputs), dim=1) # bs, 2*hidden_size

            weight = self.sigmoid(self.W(combined)) # bs, hidden_size

            fused = weight * entity_outputs + (1 - weight) * entity_annotation_outputs # bs, hidden_size

            merged = entity_outputs + fused
        else:
            merged = entity_outputs

        merged = self.dropout(merged)
        logits = self.classifier(merged)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else:
            logits = logits.argmax(-1)
            return loss, logits, label_ids

class Model4Membership(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()
        if "bert" in args.model_type:
            model_classes = get_model_classes()
            model_config = model_classes[args.model_type]
            self.bert = model_config['model'].from_pretrained(
                args.model_name_or_path
            )
            self.hidden_size = self.bert.config.hidden_size
        else:
            self.hidden_size = args.embedding_dim
            self.bert = nn.Embedding(len(data_processor.word2id), self.hidden_size)
            self.bert.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
            self.bert.weight.requires_grad = True
        self.model_type = args.model_type
        self.annotation_free = args.annotation_free
        self.name_label = args.name_label
        self.W = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.merge_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if args.continue_training:
            with safe_open(args.pretrained_model_path, framework="pt", device="cpu") as f:
                state_dict = {key: torch.tensor(f.get_tensor(key)) for key in f.keys()}
            bert_state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('bert')}
            self.bert.load_state_dict(bert_state_dict, strict=False)
        self.num_labels = 2
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_labels)

    def get_embedding(self, input_ids, attention_mask, model_type):
        if "bert" in model_type:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0]
        else:
            embedded = self.bert(input_ids)
            mask = attention_mask.unsqueeze(-1).expand(embedded.size()).float()
            masked_embed = embedded * mask
            sum_embeddings = torch.sum(masked_embed, dim=1)  # (bs, embedding_dim)
            num_non_pad = torch.sum(attention_mask, dim=1).unsqueeze(-1)  # (bs, 1)
            average_embed = sum_embeddings / num_non_pad  # (bs, embedding_dim)       
            return average_embed

    def forward(self, instance_input_ids, instance_attention_mask, entity_input_ids, entity_attention_mask, instance_label_input_ids, instance_label_attention_mask, entity_label_input_ids, entity_label_attention_mask, 
                 instance_annotation_input_ids, instance_annotation_attention_mask, entity_annotation_input_ids, entity_annotation_attention_mask, label_ids, mode):
        if self.name_label == "name":
            instance_outputs = self.get_embedding(input_ids=instance_input_ids, attention_mask=instance_attention_mask, model_type=self.model_type)
            entity_outputs = self.get_embedding(input_ids=entity_input_ids, attention_mask=entity_attention_mask, model_type=self.model_type)
        elif self.name_label == "label":
            instance_outputs = self.get_embedding(input_ids=instance_label_input_ids, attention_mask=instance_label_attention_mask, model_type=self.model_type)
            entity_outputs = self.get_embedding(input_ids=entity_label_input_ids, attention_mask=entity_label_attention_mask, model_type=self.model_type)
        else:
            instance_outputs = self.get_embedding(input_ids=instance_input_ids, attention_mask=instance_attention_mask, model_type=self.model_type)
            entity_outputs = self.get_embedding(input_ids=entity_input_ids, attention_mask=entity_attention_mask, model_type=self.model_type)
            instance_label_outputs = self.get_embedding(input_ids=instance_label_input_ids, attention_mask=instance_label_attention_mask, model_type=self.model_type)
            entity_label_outputs = self.get_embedding(input_ids=entity_label_input_ids, attention_mask=entity_label_attention_mask, model_type=self.model_type)
            instance_outputs = instance_outputs + instance_label_outputs
            entity_outputs = entity_outputs + entity_label_outputs
        
        if not self.annotation_free:
            instance_annotation_outputs = self.get_embedding(input_ids=instance_annotation_input_ids, attention_mask=instance_annotation_attention_mask, model_type=self.model_type)
            entity_annotation_outputs = self.get_embedding(input_ids=entity_annotation_input_ids, attention_mask=entity_annotation_attention_mask, model_type=self.model_type)

            instance_combined = torch.cat((instance_outputs, instance_annotation_outputs), dim=1) # bs, 2*hidden_size
            entity_combined = torch.cat((entity_outputs, entity_annotation_outputs), dim=1) # bs, 2*hidden_size

            instance_weight = self.sigmoid(self.W(instance_combined)) # bs, hidden_size
            entity_weight = self.sigmoid(self.W(entity_combined)) # bs, hidden_size

            instance_fused = instance_weight * instance_outputs + (1 - instance_weight) * instance_annotation_outputs # bs, hidden_size
            entity_fused = entity_weight * entity_outputs + (1 - entity_weight) * entity_annotation_outputs # bs, hidden_size
            instance_merged = instance_outputs + instance_fused
            entity_merged = entity_outputs + entity_fused
        else:
            instance_merged = instance_outputs
            entity_merged = entity_outputs

        sequence_output = torch.cat((instance_merged, entity_merged), dim=1) # bs, 2*hidden_size
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else:
            logits = logits.argmax(-1)
            return loss, logits, label_ids
