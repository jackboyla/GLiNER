from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
import os

def generate_entity_pairs_indices(self, span_idx):
    num_entities = span_idx.size(0)  # [num_ents, 2]

    # Expand and tile to create all possible pairs
    span_idx_expanded = span_idx.unsqueeze(1).expand(-1, num_entities, -1)  #  ([num_entities, num_entities, 2])
    span_idx_tiled = span_idx.unsqueeze(0).expand(num_entities, -1, -1)     #  ([num_entities, num_entities, 2])

    # pair_reps = torch.cat([span_idx_expanded, span_idx_tiled], dim=2)

    # Now, we need to filter out the self-pairs and duplicates. We use an upper triangular mask for this
    triu_mask = torch.triu(torch.ones(num_entities, num_entities), diagonal=1).bool()  #  ([ num_entities, num_entities ])

    # num_unique_pairs is the total number of unique pairs that can be formed from num_entities, excluding self-pairs
    combined_pairs = torch.stack((span_idx_expanded[triu_mask], span_idx_tiled[triu_mask]), dim=1)  

    return combined_pairs  #  ([num_unique_pairs, 2 ->start_index, 2 ->end_index])


class InstructBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_width = config.max_width
        self.base_config = config

    def get_dict(self, spans, classes_to_id):
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag
    
    def get_rel_labels(self, relations_idx, relations, rel_label_dict):
        # get the class for each relation pair
        labelled_rel_dict = {
            (tuple(rel['head']['position']), tuple(rel['tail']['position'])) : rel['relation_text'] for rel in relations
        }
        relations_idx = relations_idx.tolist()

        # assign label if in dataset else -1
        rel_labels = []
        for rel in relations_idx:
            head_idx, tail_idx = tuple(rel[0]), tuple(rel[1])
            if (head_idx, tail_idx) in labelled_rel_dict:
                label = rel_label_dict[labelled_rel_dict[(head_idx, tail_idx)]]
                rel_labels.append(label)
            else:
                rel_labels.append(0)
        
        return rel_labels
    

    def preprocess_spans(self, tokens, ner, classes_to_id, relations):

        max_len = self.base_config.max_len

        if len(tokens) > max_len:
            seq_length = max_len
            tokens = tokens[:max_len]
        else:
            seq_length = len(tokens)

        spans_idx = []

        if os.environ.get('TASK') == 'ner':
            for i in range(seq_length):
                spans_idx.extend([(i, i + j) for j in range(self.max_width)])
            dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)  # {(0, 5): 1, (0, 0): 0, (0, 1): 0, ...}

            # 0 for null labels
            span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])  # [num_possible_spans]
            spans_idx = torch.LongTensor(spans_idx)                          # [num_possible_spans, 2]
            
            # mask for valid spans
            valid_span_mask = spans_idx[:, 1] > seq_length - 1

            # mask invalid positions
            span_label = span_label.masked_fill(valid_span_mask, -1)  # [num_possible_spans]


        elif os.environ.get('TASK') == 'rel':
            for ner_span in ner:
                start, end = ner_span[0], ner_span[1]
                spans_idx.append((start, end))
            
            spans_idx = torch.LongTensor(spans_idx)                        # [num_possible_spans, 2]
            relations_idx = generate_entity_pairs_indices(spans_idx)  # [num_ent_pairs, 2, 2]

            # TODO: make sure model generates relations in the same way as generate_entity_pairs_indices()

            unique_rel_labels = set([rel['relation_text'] for rel in relations])
            rel_label_dict = {r: (i+1) for i, r in enumerate(unique_rel_labels)}
            # 0 for null labels
            rel_label = torch.LongTensor(self.get_rel_labels(relations_idx, relations, rel_label_dict))  # [num_ent_pairs]

            # mask for valid spans
            valid_span_mask = spans_idx[:, 1] > seq_length - 1

            # mask invalid positions
            span_label = torch.ones(spans_idx.size(0), dtype=torch.long)
            span_label = span_label.masked_fill(valid_span_mask, -1)  # [num_possible_spans]


        out = {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': seq_length,
            'entities': ner,
            'rel_label': rel_label if 'rel_label' in locals() else None,
        }
        return out

    def collate_fn(self, batch_list, entity_types=None):
        # batch_list: list of dict containing tokens, ner
        if entity_types is None:
            if os.environ['TASK'] == 'ner':
                negs = self.get_negatives(batch_list, 100)
            elif os.environ['TASK'] == 'rel':
                # TODO: add negatives for relations
                negs = self.get_negatives_rel(batch_list, 100)
            class_to_ids = []
            id_to_classes = []
            for b in batch_list:
                # negs = b["negative"]
                random.shuffle(negs)

                # negs = negs[:sampled_neg]
                max_neg_type_ratio = int(self.base_config.max_neg_type_ratio)

                if max_neg_type_ratio == 0:
                    # no negatives
                    neg_type_ratio = 0
                else:
                    neg_type_ratio = random.randint(0, max_neg_type_ratio)

                if neg_type_ratio == 0:
                    # no negatives
                    negs_i = []
                else:
                    if os.environ['TASK'] == 'ner':
                        negs_i = negs[:len(b['ner']) * neg_type_ratio]
                    elif os.environ['TASK'] == 'rel':
                        negs_i = negs[:len(b['relations']) * neg_type_ratio]
                    
                if os.environ['TASK'] == 'ner':
                    # this is the list of all possible entity types (positive and negative)
                    types = list(set([el[2] for el in b['ner']] + negs_i))
                elif os.environ['TASK'] == 'rel':
                    types = list(set([el['relation_text'] for el in b['relations']] + negs_i))


                # shuffle (every epoch)
                random.shuffle(types)

                if len(types) != 0:
                    # prob of higher number shoul
                    # random drop
                    if self.base_config.random_drop:
                        num_ents = random.randint(1, len(types))
                        types = types[:num_ents]

                # maximum number of entities types
                types = types[:int(self.base_config.max_types)]

                # supervised training
                if "label" in b:
                    types = sorted(b["label"])

                class_to_id = {k: v for v, k in enumerate(types, start=1)}
                id_to_class = {k: v for v, k in class_to_id.items()}
                class_to_ids.append(class_to_id)
                id_to_classes.append(id_to_class)

            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids[i], b.get('relations')) for i, b in enumerate(batch_list)
            ]

        else:
            class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
            id_to_classes = {k: v for v, k in class_to_ids.items()}
            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids, b.get('relations')) for b in batch_list
            ]

        span_idx = pad_sequence(
            [b['span_idx'] for b in batch], batch_first=True, padding_value=0
        )

        span_label = pad_sequence(
            [el['span_label'] for el in batch], batch_first=True, padding_value=-1
        )

        if os.environ['TASK'] == 'rel':
            rel_label = pad_sequence(
                [el['rel_label'] for el in batch], batch_first=True, padding_value=-1
            )
        

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_label != -1,
            'span_label': span_label,
            'rel_label': rel_label if 'rel_label' in locals() else None,
            'entities': [el['entities'] for el in batch],
            'classes_to_id': class_to_ids,
            'id_to_classes': id_to_classes,
        }

    @staticmethod
    def get_negatives(batch_list, sampled_neg=5):
        ent_types = []
        for b in batch_list:
            types = set([el[2] for el in b['ner']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        # sample negatives
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]
    
    @staticmethod
    def get_negatives_rel(batch_list, sampled_neg=5):
        rel_types = []
        for b in batch_list:
            types = set([el['relation_text'] for el in b['relations']])
            rel_types.extend(list(types))
        rel_types = list(set(rel_types))
        # sample negatives
        random.shuffle(rel_types)
        return rel_types[:sampled_neg]

    def create_dataloader(self, data, entity_types=None, **kwargs):
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), **kwargs)

    def set_sampling_params(self, max_types, shuffle_types, random_drop, max_neg_type_ratio, max_len):
        """
        Sets sampling parameters on the given model.

        Parameters:
        - model: The model object to update.
        - max_types: Maximum types parameter.
        - shuffle_types: Boolean indicating whether to shuffle types.
        - random_drop: Boolean indicating whether to randomly drop elements.
        - max_neg_type_ratio: Maximum negative type ratio.
        - max_len: Maximum length parameter.
        """
        self.base_config.max_types = max_types
        self.base_config.shuffle_types = shuffle_types
        self.base_config.random_drop = random_drop
        self.base_config.max_neg_type_ratio = max_neg_type_ratio
        self.base_config.max_len = max_len
