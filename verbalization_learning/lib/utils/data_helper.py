import os
import json
import math
import numpy as np
import pickle
import random
from tqdm import tqdm
import copy

import torch
from torch.utils.data import TensorDataset


class Data_Helper(object):
    """docstring for Data_Helper"""
    def __init__(self, tokenizer, args, split='test', inference=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.args = args

        self.data_dir = os.path.join('./data', args.dataset)

        from .data_processor import RawDataProcessor
        self.data_collator = self.data_collator_for_concept2sentence
        self.processor = RawDataProcessor(self.data_dir, tokenizer, args)

        if inference:
            self.testset = self.processor.load_datasplit(split)
        else:
            self.trainset = self.processor.load_trainset()
            self.devset = self.processor.load_devset()

            if args.graph_source_alpha > 0:
                self.trainset_with_groundtruth = self.processor.load_datasplit('train.groundtruth')

    def sequential_iterate(self, dataset, batch_size):
        data_size = len(dataset)
        batch_num = math.ceil(data_size / batch_size)
                 
        for batch_id in range(batch_num):
            start_index = batch_id * batch_size
            end_index = min((batch_id+1) * batch_size, data_size)
            batch = dataset[start_index:end_index]
            yield batch

    def relation_sampling(self, graph):
        if self.args.sample_ratio == 0:
            return graph

        if len(graph) == 0:
            return graph

    def data_collator_for_concept2sentence_inference(self, features):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature in features:
            input_seq = self.processor.format_input(feature.context, feature.entities, feature.relations)
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])

    def data_collator_for_graph2story_inference(self, features, batch_generated_context, batch_generated_graph):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature, generated_context, generated_graph in zip(features, batch_generated_context, batch_generated_graph):
            input_seq = 'context: ' + generated_context + ' entities: ' +  '<SEP>'.join(feature.entities) + ' relations: ' + generated_graph
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])

    def data_collator_for_concept2graph_inference(self, tokenizer, features, batch_generated_context):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature, generated_context in zip(features, batch_generated_context):
            input_seq = 'context: ' + generated_context + ' entities: ' +  '<SEP>'.join(feature.entities)
            encoder_input = tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])

    def data_collator_for_concept2sentence(self, features):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        text_label_tensor = []

        for feature in features:
            if self.args.mask_ratio > 0:
                max_num_mask = int(len(feature.entities) * self.args.mask_ratio)
                num_mask = random.choice(range(max_num_mask + 1))
                input_entities = random.sample(feature.entities, len(feature.entities) - num_mask)
            else:
                input_entities = random.sample(feature.entities, len(feature.entities))

            relations = self.relation_sampling(feature.relations)
            input_seq = self.processor.format_input(feature.context, input_entities, relations)
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

            text_label_tensor.append(feature.text_label)

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor,
                                            text_label_tensor])

    def _get_tensor_graph_input(self, relations):

        sequence = ''
        if len(relations) == 0:
            return sequence
        triplet = relations[0]
        sequence += triplet[0] + '<{}>'.format(triplet[1]) + triplet[2]

        for triplet in relations[1:]:
            sequence += '<SEP>' + triplet[0] + '<{}>'.format(triplet[1]) + triplet[2]
        return sequence
        
