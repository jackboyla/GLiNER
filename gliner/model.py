import argparse
import json
from pathlib import Path
import re
import os
from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F
import yaml
from gliner.modules.layers import LstmSeq2SeqEncoder
from gliner.modules.base import InstructBase
from gliner.modules.evaluator import Evaluator, greedy_search, RelEvaluator
from gliner.modules.span_rep import SpanRepLayer
from gliner.modules.rel_rep import RelRepLayer
from gliner.modules.token_rep import TokenRepLayer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])



class GLiNER(InstructBase, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # [ENT] token
        self.entity_token = "<<ENT>>"
        self.sep_token = "<<SEP>>"

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=[self.entity_token, self.sep_token])

        # hierarchical representation of tokens (zaratiana et al, 2022)
        # https://arxiv.org/pdf/2203.14710.pdf
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        if os.environ.get('TASK') == 'ner':
            # span representation
            # we have a paper to study span representation for ner
            # zaratiana et al, 2022: https://aclanthology.org/2022.umios-1.1/
            self.span_rep_layer = SpanRepLayer(
                span_mode=config.span_mode,
                hidden_size=config.hidden_size,
                max_width=config.max_width,
                dropout=config.dropout,
            )
        elif os.environ.get('TASK') == 'rel':
            self.span_rep_layer = RelRepLayer(
                rel_mode=config.span_mode,
                hidden_size=config.hidden_size,
                max_width=config.max_width,
                dropout=config.dropout,
            )

        # prompt representation (FFN)
        self.prompt_rep_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

    def get_optimizer(self, lr_encoder, lr_others, freeze_token_rep=False):
        """
        Parameters:
        - lr_encoder: Learning rate for the encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: whether the token representation layer should be frozen.
        """
        param_groups = [
            # encoder
            {'params': self.rnn.parameters(), 'lr': lr_others},
            # projection layers
            {'params': self.span_rep_layer.parameters(), 'lr': lr_others},
            {'params': self.prompt_rep_layer.parameters(), 'lr': lr_others},
        ]

        if not freeze_token_rep:
            # If token_rep_layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({'params': self.token_rep_layer.parameters(), 'lr': lr_encoder})
        else:
            # If token_rep_layer should be frozen, explicitly set requires_grad to False for its parameters
            for param in self.token_rep_layer.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups)

        return optimizer

    def compute_score_train(self, x):

        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)  # ([B, num_possible_spans, 2])  *  ([B, num_possible_spans, 1])

        new_length = x['seq_length'].clone()
        new_tokens = []
        all_len_prompt = []
        num_classes_all = []

        # add prompt to the tokens
        for i in range(len(x['tokens'])):
            all_types_i = list(x['classes_to_id'][i].keys())
            # multiple entity types in all_types. Prompt is appended at the start of tokens
            entity_prompt = []
            num_classes_all.append(len(all_types_i))
            # add enity types to prompt
            for entity_type in all_types_i:
                entity_prompt.append(self.entity_token)  # [ENT] token
                entity_prompt.append(entity_type)        # entity type
            entity_prompt.append(self.sep_token)         # [SEP] token

            # prompt format:
            # [ENT] entity_type [ENT] entity_type ... [ENT] entity_type [SEP]

            # add prompt to the tokens
            tokens_p = entity_prompt + x['tokens'][i]

            # input format:
            # [ENT] entity_type_1 [ENT] entity_type_2 ... [ENT] entity_type_m [SEP] token_1 token_2 ... token_n

            # update length of the sequence (add prompt length to the original length)
            new_length[i] = new_length[i] + len(entity_prompt)
            # update tokens
            new_tokens.append(tokens_p)
            # store prompt length
            all_len_prompt.append(len(entity_prompt))

        # create a mask using num_classes_all (False, if it exceeds the number of classes, True otherwise)
        max_num_classes = max(num_classes_all)
        entity_type_mask = torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes_all), -1).to(
            x['span_mask'].device)
        entity_type_mask = entity_type_mask < torch.tensor(num_classes_all).unsqueeze(-1).to(
            x['span_mask'].device)  # [batch_size, max_num_classes]

        # compute all token representations
        bert_output = self.token_rep_layer(new_tokens, new_length)
        word_rep_w_prompt = bert_output["embeddings"]  # embeddings for all tokens (with prompt)
        mask_w_prompt = bert_output["mask"]  # mask for all tokens (with prompt)

        # get word representation (after [SEP]), mask (after [SEP]) and entity type representation (before [SEP])
        word_rep = []  # word representation (after [SEP])
        mask = []  # mask (after [SEP])
        entity_type_rep = []  # entity type representation (before [SEP])
        for i in range(len(x['tokens'])):
            prompt_entity_length = all_len_prompt[i]  # length of prompt for this example
            # get word representation (after [SEP])
            word_rep.append(word_rep_w_prompt[i, prompt_entity_length:prompt_entity_length + x['seq_length'][i]])
            # get mask (after [SEP])
            mask.append(mask_w_prompt[i, prompt_entity_length:prompt_entity_length + x['seq_length'][i]])

            # get entity type representation (before [SEP])
            entity_rep = word_rep_w_prompt[i, :prompt_entity_length - 1]  # remove [SEP]
            entity_rep = entity_rep[0::2]  # it means that we take every second element starting from the second one
            entity_type_rep.append(entity_rep)

        # padding for word_rep, mask and entity_type_rep
        word_rep = pad_sequence(word_rep, batch_first=True)                # [batch_size, seq_len, hidden_size]
        mask = pad_sequence(mask, batch_first=True)                        # [batch_size, seq_len]
        entity_type_rep = pad_sequence(entity_type_rep, batch_first=True)  # [batch_size, len_types, hidden_size]

        # compute span representation
        word_rep = self.rnn(word_rep, mask)                  # ([B, seq_length, D])
        span_rep = self.span_rep_layer(word_rep, span_idx)   # NER --> ([B, seq_length, 12, D]) 
                                                             # or REL --> ([B, num_pairs, D])

        # compute final entity type representation (FFN)
        entity_type_rep = self.prompt_rep_layer(entity_type_rep)  # (batch_size, len_types, hidden_size)
        num_classes = entity_type_rep.shape[1]                    # number of entity types

        # similarity score
        if os.environ.get('TASK') == 'ner':
            scores = torch.einsum('BLKD,BCD->BLKC', span_rep, entity_type_rep) # ([B, L, K, num_classes])
        elif os.environ.get('TASK') == 'rel':
            scores = torch.einsum('BKD,BCD->BKC', span_rep, entity_type_rep) # ([B, num_pairs, num_classes])

        return scores, num_classes, entity_type_mask   #  see above, num_classes, ([B, num_classes])

    def forward(self, x):
        # compute span representation
        scores, num_classes, entity_type_mask = self.compute_score_train(x)
        batch_size = scores.shape[0]


        # loss for filtering classifier
        logits_label = scores.view(-1, num_classes)

        if os.environ['TASK'] == 'ner':
            labels = x["span_label"].view(-1)    # [B * num_spans]
        elif os.environ['TASK'] == 'rel':
            labels = x['rel_label'].view(-1)     # [B * num_entity_pairs]

        mask_label = labels != -1
        labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

        # one-hot encoding
        labels_one_hot = torch.zeros(labels.size(0), num_classes + 1, dtype=torch.float32).to(scores.device) # ([batch_size * num_spans, num_classes + 1])
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Set the corresponding index to 1
        labels_one_hot = labels_one_hot[:, 1:]              # Remove the first column
        # Shape of labels_one_hot: (batch_size * num_spans, num_classes)

        # compute loss (without reduction)
        all_losses = F.binary_cross_entropy_with_logits(logits_label, labels_one_hot,
                                                        reduction='none')
        # mask loss using entity_type_mask (B, C)
        masked_loss = all_losses.view(batch_size, -1, num_classes) * entity_type_mask.unsqueeze(1)   #  ([B, L*K, num_classes])  *  ([B, 1, num_classes])
        all_losses = masked_loss.view(-1, num_classes)
        # expand mask_label to all_losses
        mask_label = mask_label.unsqueeze(-1).expand_as(all_losses)
        # put lower loss for in label_one_hot (2 for positive, 1 for negative)
        weight_c = labels_one_hot + 1
        # apply mask
        all_losses = all_losses * mask_label.float() * weight_c
        loss = all_losses.sum()

        return loss


    def compute_score_eval(self, x, device):
        # check if classes_to_id is dict
        assert isinstance(x['classes_to_id'], dict), "classes_to_id must be a dict"

        span_idx = (x['span_idx'] * x['span_mask'].unsqueeze(-1)).to(device)

        all_types = list(x['classes_to_id'].keys())
        # multiple entity types in all_types. Prompt is appended at the start of tokens
        entity_prompt = []

        # add enity types to prompt
        for entity_type in all_types:
            entity_prompt.append(self.entity_token)
            entity_prompt.append(entity_type)

        entity_prompt.append(self.sep_token)

        prompt_entity_length = len(entity_prompt)

        # add prompt
        tokens_p = [entity_prompt + tokens for tokens in x['tokens']]
        seq_length_p = x['seq_length'] + prompt_entity_length

        out = self.token_rep_layer(tokens_p, seq_length_p)

        word_rep_w_prompt = out["embeddings"]
        mask_w_prompt = out["mask"]

        # remove prompt
        word_rep = word_rep_w_prompt[:, prompt_entity_length:, :]
        mask = mask_w_prompt[:, prompt_entity_length:]

        # get_entity_type_rep
        entity_type_rep = word_rep_w_prompt[:, :prompt_entity_length - 1, :]
        # extract [ENT] tokens (which are at even positions in entity_type_rep)
        entity_type_rep = entity_type_rep[:, 0::2, :]

        entity_type_rep = self.prompt_rep_layer(entity_type_rep)  # (batch_size, len_types, hidden_size)

        word_rep = self.rnn(word_rep, mask)
        span_rep = self.span_rep_layer(word_rep, span_idx)

        # scores
        if os.environ['TASK'] == 'ner':
            local_scores = torch.einsum('BLKD,BCD->BLKC', span_rep, entity_type_rep) # ([B, L, K, num_classes])
        
        elif os.environ['TASK'] == 'rel':
            local_scores = torch.einsum('BKD,BCD->BKC', span_rep, entity_type_rep) # ([B, num_pairs, num_classes])
        

        return local_scores


    @torch.no_grad()
    def predict(self, x, flat_ner=False, threshold=0.5, ner=None):
        self.eval()
        local_scores = self.compute_score_eval(x, device=next(self.parameters()).device)

        if os.environ['TASK'] == 'ner':
            spans = []
            for i, _ in enumerate(x["tokens"]):
                local_i = local_scores[i]
                wh_i = [i.tolist() for i in torch.where(torch.sigmoid(local_i) > threshold)]
                span_i = []
                for s, k, c in zip(*wh_i):
                    if s + k < len(x["tokens"][i]):  # checks whether the end of a potential entity is within the sentence
                        span_i.append((s, s + k, x["id_to_classes"][c + 1], local_i[s, k, c]))
                span_i = greedy_search(span_i, flat_ner)
                spans.append(span_i)
            return spans
        
        elif os.environ['TASK'] == 'rel':

            assert isinstance(ner, list), "ner should be a list of list of spans like [[(1, 2, 'PER'), (3, 4, 'ORG'), ...], ]"

            rels = []
            for i, _ in enumerate(x["tokens"]):
                local_i = local_scores[i]  # Predictions for the i-th item in the batch
                # shape ([num_pairs, num_classes])
                probabilities = torch.sigmoid(local_i)  # Convert logits to probabilities

                # Iterate over all possible pairs and relation types
                triggered_relations = [i.tolist() for i in torch.where(probabilities > threshold)]
                # triggered_relations --> tuple of two lists, 
                # one for pair_idx * num_triggered_classes (based on threshold) 
                # and one for the corresponding tirggered rel_type_id, e.g pair [3, 3, 3] have rel type [0, 4, 5]
                rels_i = []
                for pair_idx, rel_type_idx in zip(*triggered_relations):

                    # Check if the pair index is within the bounds of the entity pairs
                    if pair_idx < len(x["relations_idx"][i]):

                        score = probabilities[pair_idx, rel_type_idx].item()
                        # Get the entity pair and relation type
                        entity_pair = x["relations_idx"][i][pair_idx] 
                        relation_type = x["id_to_classes"][rel_type_idx + 1]
                    
                        rels_i.append((entity_pair.cpu().numpy().tolist(), relation_type, score))
                
                rels.append(rels_i)
            return rels

    def predict_entities(self, text, labels, flat_ner=True, threshold=0.5, ner=None):
        return self.batch_predict_entities([text], labels, flat_ner=flat_ner, threshold=threshold, ner=[ner])[0]

    def batch_predict_entities(self, texts, labels, flat_ner=True, threshold=0.5, ner=None):
        """
        Predict entities for a batch of texts.
        texts:  List of texts | List[str]
        labels: List of labels | List[str]
        ...
        """

        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            if type(text) is str:
                for match in re.finditer(r'\w+(?:[-_]\w+)*|\S', text):
                    tokens.append(match.group())
                    start_token_idx_to_text_idx.append(match.start())
                    end_token_idx_to_text_idx.append(match.end())
            else:
                tokens = text  # already tokenized
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        if ner is not None:
            for i, x in enumerate(input_x):
                x['ner'] = ner[i]

        x = self.collate_fn(input_x, labels)
        
        outputs = self.predict(x, flat_ner=flat_ner, threshold=threshold, ner=ner)

        if os.environ['TASK'] == 'ner':
            all_entities = []
            for i, output in enumerate(outputs):
                start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
                end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
                entities = []
                for start_token_idx, end_token_idx, ent_type in output:
                    start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                    end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                    entities.append({
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                    })
                all_entities.append(entities)

            return all_entities
        
        elif os.environ['TASK'] == 'rel':
            all_relations = []

            rels = []
            for i, output in enumerate(outputs):

                
                for (head_pos, tail_pos), pred_label, score in output:

                    rel = {
                        'head_pos' : head_pos,
                        'tail_pos' : tail_pos,
                        'head_text' : texts[i][head_pos[0]:head_pos[1]],
                        'tail_text' : texts[i][tail_pos[0]:tail_pos[1]],
                        'label': pred_label,
                        'score': score,
                    }
                    
                    rels.append(rel)

                all_relations.append(rels)

            return all_relations


    def evaluate(self, test_data, flat_ner=False, threshold=0.5, batch_size=12, entity_types=None, top_k=1):
        self.eval()
        logger.info(f"Number of classes to evaluate with --> {len(entity_types)}")
        data_loader = self.create_dataloader(test_data, batch_size=batch_size, entity_types=entity_types, shuffle=False)
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in data_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            x['classes_to_id'] = x['classes_to_id'][0] if type(x['classes_to_id']) is list else x['classes_to_id']
            x['id_to_classes'] = x['id_to_classes'][0] if type(x['id_to_classes']) is list else x['id_to_classes']
            ner = x['entities']


            batch_predictions = self.predict(x, flat_ner, threshold, ner)

            if os.environ['TASK'] == 'ner':
                batch_predictions_formatted = batch_predictions
                all_trues.extend(x["entities"])
            elif os.environ['TASK'] == 'rel':
                # TODO: test throroughly
                all_trues.extend(x["relations"])
                # format relation predictions for metrics calculation
                batch_predictions_formatted = []
                for i, output in enumerate(batch_predictions):

                    # sort output by score
                    output = sorted(output, key=lambda x: x[2], reverse=True)

                    rels = []
                    position_set = {}  # track all position predictions to take top_k predictions
                    for (head_pos, tail_pos), pred_label, score in output:

                        hashable_positions = (tuple(head_pos), tuple(tail_pos))
                        if hashable_positions not in position_set:
                            position_set[hashable_positions] = 0

                        if position_set[hashable_positions] < top_k:

                            rel = {
                                'head' : {'position': head_pos},
                                'tail' : {'position': tail_pos},
                                'relation_text': pred_label,
                                'score': score,
                            }
                            
                            rels.append(rel)
                            position_set[hashable_positions] += 1

                    batch_predictions_formatted.append(rels)

            all_preds.extend(batch_predictions_formatted)
                

        if os.environ['TASK'] == 'ner':
            evaluator = Evaluator(all_trues, all_preds)
            out, f1 = evaluator.evaluate()

        elif os.environ['TASK'] == 'rel':
            evaluator = RelEvaluator(all_trues, all_preds)
            out, f1 = evaluator.evaluate()
        
        return out, f1


    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        # 1. Backwards compatibility: Use "gliner_base.pt" and "gliner_multi.pt" with all data
        filenames = ["gliner_base.pt", "gliner_multi.pt"]
        for filename in filenames:
            model_file = Path(model_id) / filename
            if not model_file.exists():
                try:
                    model_file = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                except HfHubHTTPError:
                    continue
            dict_load = torch.load(model_file, map_location=torch.device(map_location))
            config = dict_load["config"]
            state_dict = dict_load["model_weights"]
            config.model_name = "microsoft/deberta-v3-base" if filename == "gliner_base.pt" else "microsoft/mdeberta-v3-base"
            model = cls(config)
            model.load_state_dict(state_dict, strict=strict)
            # Required to update flair's internals as well:
            model.to(map_location)
            return model

        # 2. Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_file = Path(model_id) / "pytorch_model.bin"
        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config_file = Path(model_id) / "gliner_config.json"
        if not config_file.exists():
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="gliner_config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config = load_config_as_namespace(config_file)
        model = cls(config)
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict, assign=True)
        model.to(map_location)
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            if isinstance(config, argparse.Namespace):
                config = vars(config)
            (save_directory / "gliner_config.json").write_text(json.dumps(config, indent=2))

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def to(self, device):
        super().to(device)
        import flair

        flair.device = device
        return self


def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
