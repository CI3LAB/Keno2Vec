import pandas as pd
from utils.data_utils import InputExample_subsumption, InputExample_property, InputExample_multiple, InputExample_membership
import os
import json
import random
import re
from utils.w2v_utils import load_embedding

def get_label(cls_name, onto):
    if not onto[cls_name]:
        return cls_name
    cls = onto[cls_name]
    if cls.label:
        return cls.label[0]
    else:
        if cls.name:
            return cls.name
        else:
            return cls_name

class DataProcessor_subsumption:
    def __init__(self, args):
        self.data_path = args.data_dir
        self.annotation_file = args.annotation_file
        self.annotation_free = args.annotation_free
        self.ontology = args.ontology
        self.all_labels = ["0", "1"]
        label2idx = {tag: idx for idx, tag in enumerate(self.all_labels)}
        idx2label = {idx: tag for idx, tag in enumerate(self.all_labels)}
        self.label2idx = label2idx
        self.idx2label = idx2label
        if "bert" not in args.model_type:
            vec_mat, word2id, id2word=load_embedding(args.model_name_or_path)
            self.vec_mat = vec_mat
            self.word2id = word2id
            self.id2word = id2word
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]
        for index, item in enumerate(data):
            id = index
            entity_1 = item[0]
            entity_2 = item[1]
            entity_1_label = get_label(entity_1, self.ontology)
            entity_2_label = get_label(entity_2, self.ontology)
            label = self.label2idx[item[2]]
            if self.annotation_free:
                entity_1_annotation = "None"
                entity_2_annotation = "None"
            else:
                entity_1_annotation = annotations[entity_1] if entity_1 in annotations else "None"
                entity_2_annotation = annotations[entity_2] if entity_2 in annotations else "None"
                # entity_1_annotation = "None"
                # entity_2_annotation = "None"
                # entity_1_annotation = random.choice(list(annotations.values()))
                # entity_2_annotation = random.choice(list(annotations.values()))
            example = InputExample_subsumption(guid=str(id), entity_1=entity_1, entity_2=entity_2, entity_1_label=entity_1_label, entity_2_label=entity_2_label, label=label, entity_1_annotation=entity_1_annotation, entity_2_annotation=entity_2_annotation)
            examples.append(example)
        return examples
    
    def get_examples_sample(self, sample_ratio, seed, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]

        label2item = {d:[] for d in self.all_labels}

        for item in data:
            label2item[item[2]].append(item)
        sample_num = int(len(data) * sample_ratio / len(label2item))
        for label in label2item:
            random.seed(seed)
            samples = random.sample(label2item[label], sample_num)
            for index, item in enumerate(samples):
                id = index
                entity_1 = item[0]
                entity_2 = item[1]
                entity_1_label = get_label(entity_1, self.ontology)
                entity_2_label = get_label(entity_2, self.ontology)
                label = self.label2idx[item[2]]
                if self.annotation_free:
                    entity_1_annotation = "None"
                    entity_2_annotation = "None"
                else:
                    entity_1_annotation = annotations[entity_1] if entity_1 in annotations else "None"
                    entity_2_annotation = annotations[entity_2] if entity_2 in annotations else "None"
                example = InputExample_subsumption(guid=str(id), entity_1=entity_1, entity_2=entity_2, entity_1_label=entity_1_label, entity_2_label=entity_2_label, label=label, entity_1_annotation=entity_1_annotation, entity_2_annotation=entity_2_annotation)
                examples.append(example)
        
        return examples

class DataProcessor_property:
    def __init__(self, args):
        self.data_path = args.data_dir
        self.annotation_file = args.annotation_file
        self.annotation_free = args.annotation_free
        self.ontology = args.ontology
        self.all_labels = ["0", "1"]
        label2idx = {tag: idx for idx, tag in enumerate(self.all_labels)}
        idx2label = {idx: tag for idx, tag in enumerate(self.all_labels)}
        self.label2idx = label2idx
        self.idx2label = idx2label
        if "bert" not in args.model_type:
            vec_mat, word2id, id2word=load_embedding(args.model_name_or_path)
            self.vec_mat = vec_mat
            self.word2id = word2id
            self.id2word = id2word
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]
        for index, item in enumerate(data):
            id = index
            entity_1 = item[0]
            property = item[1]
            entity_2 = item[2]
            entity_1_label = get_label(entity_1, self.ontology)
            property_label = get_label(property, self.ontology)
            entity_2_label = get_label(entity_2, self.ontology)
            label = self.label2idx[item[3]]
            if self.annotation_free:
                entity_1_annotation = "None"
                property_annotation = "None"
                entity_2_annotation = "None"
            else:
                entity_1_annotation = annotations[entity_1] if entity_1 in annotations else "None"
                property_annotation = annotations[property] if property in annotations else "None"
                entity_2_annotation = annotations[entity_2] if entity_2 in annotations else "None"
                # entity_1_annotation = "None"
                # property_annotation = "None"
                # entity_2_annotation = "None"
                # entity_1_annotation = random.choice(list(annotations.values()))
                # property_annotation = random.choice(list(annotations.values()))
                # entity_2_annotation = random.choice(list(annotations.values()))
            example = InputExample_property(guid=str(id), entity_1=entity_1, property=property, entity_2=entity_2, entity_1_label=entity_1_label, property_label=property_label, entity_2_label=entity_2_label,
                                            label=label, entity_1_annotation=entity_1_annotation, property_annotation=property_annotation, entity_2_annotation=entity_2_annotation)
            examples.append(example)
        return examples

    def get_examples_sample(self, sample_ratio, seed, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]

        label2item = {d:[] for d in self.all_labels}

        for item in data:
            label2item[item[3]].append(item)
        sample_num = int(len(data) * sample_ratio / len(label2item))
        for label in label2item:
            random.seed(seed)
            samples = random.sample(label2item[label], sample_num)
            for index, item in enumerate(samples):
                id = index
                entity_1 = item[0]
                property = item[1]
                entity_2 = item[2]
                entity_1_label = get_label(entity_1, self.ontology)
                property_label = get_label(property, self.ontology)
                entity_2_label = get_label(entity_2, self.ontology)
                label = self.label2idx[item[3]]
                if self.annotation_free:
                    entity_1_annotation = "None"
                    property_annotation = "None"
                    entity_2_annotation = "None"
                else:
                    entity_1_annotation = annotations[entity_1] if entity_1 in annotations else "None"
                    property_annotation = annotations[property] if property in annotations else "None"
                    entity_2_annotation = annotations[entity_2] if entity_2 in annotations else "None"
                example = InputExample_property(guid=str(id), entity_1=entity_1, property=property, entity_2=entity_2, entity_1_label=entity_1_label, property_label=property_label, entity_2_label=entity_2_label,
                                                label=label, entity_1_annotation=entity_1_annotation, property_annotation=property_annotation, entity_2_annotation=entity_2_annotation)
                examples.append(example)
        
        return examples

class DataProcessor_multiple:
    def __init__(self, args):
        self.data_path = args.data_dir
        self.annotation_file = args.annotation_file
        self.annotation_free = args.annotation_free
        self.ontology = args.ontology
        if args.multiple == "category":
            self.all_labels = ['Other', 'Entities', 'Types']
        elif args.multiple == "schema":
            self.all_labels = ['IfcElectricalDomain', 'IfcSharedComponentElements', 'IfcProcessExtension', 'IfcArchitectureDomain', 'IfcControlExtension', 'IfcPresentationOrganizationResource', 
                               'IfcDateTimeResource', 'Other', 'IfcPresentationDefinitionResource', 'IfcApprovalResource', 'IfcMeasureResource', 'IfcGeometricModelResource', 'IfcActorResource',
                               'IfcConstructionMgmtDomain', 'IfcSharedFacilitiesElements', 'IfcUtilityResource', 'IfcSharedBldgElements', 'IfcBuildingControlsDomain', 'IfcConstraintResource', 
                               'IfcPropertyResource', 'IfcSharedBldgServiceElements', 'IfcExternalReferenceResource', 'IfcPlumbingFireProtectionDomain', 'IfcMaterialResource', 'IfcSharedMgmtElements',
                               'IfcGeometryResource', 'IfcTopologyResource', 'IfcHvacDomain', 'IfcProfileResource', 'IfcStructuralLoadResource', 'IfcKernel', 'IfcStructuralAnalysisDomain',
                               'IfcPresentationAppearanceResource', 'IfcRepresentationResource', 'IfcProductExtension', 'IfcCostResource', 'IfcStructuralElementsDomain', 'IfcGeometricConstraintResource', 'IfcQuantityResource']
        else:
            raise ValueError("Invalid multiple label type")
        label2idx = {tag: idx for idx, tag in enumerate(self.all_labels)}
        idx2label = {idx: tag for idx, tag in enumerate(self.all_labels)}
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.num_labels = len(self.all_labels)
        if "bert" not in args.model_type:
            vec_mat, word2id, id2word=load_embedding(args.model_name_or_path)
            self.vec_mat = vec_mat
            self.word2id = word2id
            self.id2word = id2word
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]
        for index, item in enumerate(data):
            id = index
            entity = item[0]
            entity_label = get_label(entity, self.ontology)
            label = self.label2idx[item[1]]
            if self.annotation_free:
                entity_annotation = "None"
            else:
                entity_annotation = annotations[entity] if entity in annotations else "None"
                # entity_annotation = "None"
                # entity_annotation = random.choice(list(annotations.values()))
            example = InputExample_multiple(guid=str(id), entity=entity, entity_label=entity_label, label=label, entity_annotation=entity_annotation)
            examples.append(example)
        return examples
    
    def get_examples_sample(self, sample_ratio, seed, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]

        label2item = {d:[] for d in self.all_labels}

        for item in data:
            label2item[item[1]].append(item)
        sample_num = int(len(data) * sample_ratio / len(label2item))
        for label in label2item:
            random.seed(seed)
            samples = random.sample(label2item[label], sample_num)
            for index, item in enumerate(samples):
                id = index
                entity = item[0]
                entity_label = get_label(entity, self.ontology)
                label = self.label2idx[item[1]]
                if self.annotation_free:
                    entity_annotation = "None"
                else:
                    entity_annotation = annotations[entity] if entity in annotations else "None"
                example = InputExample_multiple(guid=str(id), entity=entity, entity_label=entity_label, label=label, entity_annotation=entity_annotation)
                examples.append(example)
        
        return examples
    
class DataProcessor_membership:
    def __init__(self, args):
        self.data_path = args.data_dir
        self.annotation_file = args.annotation_file
        self.annotation_free = args.annotation_free
        self.ontology = args.ontology
        self.all_labels = ["0", "1"]
        label2idx = {tag: idx for idx, tag in enumerate(self.all_labels)}
        idx2label = {idx: tag for idx, tag in enumerate(self.all_labels)}
        self.label2idx = label2idx
        self.idx2label = idx2label
        if "bert" not in args.model_type:
            vec_mat, word2id, id2word=load_embedding(args.model_name_or_path)
            self.vec_mat = vec_mat
            self.word2id = word2id
            self.id2word = id2word
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]
        for index, item in enumerate(data):
            id = index
            instance = item[0]
            instance_label = get_label(instance, self.ontology)
            entity = item[1]
            entity_label = get_label(entity, self.ontology)
            label = self.label2idx[item[2]]
            if self.annotation_free:
                instance_annotation = "None"
                entity_annotation = "None"
            else:
                instance_annotation = annotations[instance] if instance in annotations else "None"
                entity_annotation = annotations[entity] if entity in annotations else "None"
                # instance_annotation = "None"
                # entity_annotation = "None"
                # instance_annotation = random.choice(list(annotations.values()))
                # entity_annotation = random.choice(list(annotations.values()))
            example = InputExample_membership(guid=str(id), instance=instance, entity=entity, instance_label=instance_label, entity_label=entity_label, label=label, instance_annotation=instance_annotation, entity_annotation=entity_annotation)
            examples.append(example)
        return examples
    
    def get_examples_sample(self, sample_ratio, seed, split=None):
        path = os.path.join(self.data_path, '{}.txt'.format(split))
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data = [d.strip().split("\t") for d in data]

        label2item = {d:[] for d in self.all_labels}

        for item in data:
            label2item[item[2]].append(item)
        sample_num = int(len(data) * sample_ratio / len(label2item))
        for label in label2item:
            random.seed(seed)
            samples = random.sample(label2item[label], sample_num)
            for index, item in enumerate(samples):
                id = index
                instance = item[0]
                instance_label = get_label(instance, self.ontology)
                entity = item[1]
                entity_label = get_label(entity, self.ontology)
                label = self.label2idx[item[2]]
                if self.annotation_free:
                    instance_annotation = "None"
                    entity_annotation = "None"
                else:
                    instance_annotation = annotations[instance] if instance in annotations else "None"
                    entity_annotation = annotations[entity] if entity in annotations else "None"
                example = InputExample_membership(guid=str(id), instance=instance, entity=entity, instance_label=instance_label, entity_label=entity_label, label=label, instance_annotation=instance_annotation, entity_annotation=entity_annotation)
                examples.append(example)
        
        return examples
