import torch
import torch.nn.functional as f
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from model import sampling
from model import util
from model import feature_enhancers as fe

from typing import List, Dict


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100,
                 feature_enhancer: str = "pass"):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers
        self.feature_enhancer = fe.get_feature_enhancer(feature_enhancer)(config.hidden_size, config.hidden_size)
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h_bert = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        # print("h_bert", h_bert.shape, h_bert[1, :, 1])
        # print("mask bert", context_mask.shape, context_mask[1, :])
        # lengths = context_mask.sum(dim=1).int().tolist()
        # print("lengths", lengths)

        # enhance hidden features
        orig_shape = h_bert.shape
        h = self.feature_enhancer.prepare_input(h_bert, context_mask)
        h = self.feature_enhancer(h)
        h = self.feature_enhancer.prepare_output(h, orig_shape)
        # print("h_fe_prepped", h[1, :, 1])


        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # classify relations
        rel_masks = rel_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor = None,
                      entity_sample_mask: torch.tensor = None):
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        # enhance hidden features
        orig_shape = h.shape
        h = self.feature_enhancer.prepare_input(h, context_mask)
        h = self.feature_enhancer(h)
        h = self.feature_enhancer.prepare_output(h, orig_shape)

        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]
        ctx_size = context_mask.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_mask, ctx_size)
        rel_masks = rel_masks.float()
        rel_sample_masks = rel_sample_masks.float()
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        # max pool entity candidate spans
        entity_spans_pool = entity_masks.unsqueeze(-1) * h.unsqueeze(1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        rel_ctx = rel_masks * h
        rel_ctx = rel_ctx.max(dim=2)[0]

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_mask, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_mask.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device).unsqueeze(-1)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device).unsqueeze(-1)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


class SpEER(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100, 
                 encoding_size: int = 200, type_key="type_index", feature_enhancer: str = "pass"):
        super(SpEER, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers
        self.encoding_size = encoding_size
        self.feature_enhancer = fe.get_feature_enhancer(feature_enhancer)(config.hidden_size, config.hidden_size)
        self.rel_encoder = nn.Linear(config.hidden_size * 3 + size_embedding * 2, encoding_size)
        self.entity_encoder = nn.Linear(config.hidden_size * 2 + size_embedding, encoding_size)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs
        self._type_key = type_key

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        # enhance hidden features
        orig_shape = h.shape
        h = self.feature_enhancer.prepare_input(h, context_mask)
        h = self.feature_enhancer(h)
        h = self.feature_enhancer.prepare_output(h, orig_shape)
        
        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]
        device = self.entity_encoder.weight.device

        # encode and classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_encoding, entity_spans_pool = self._encode_entities(encodings, h, entity_masks, size_embeddings)
        entity_clf = self._classify_entities(entity_encoding)

        # prepare relation encoding
        rel_masks = rel_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_encoding = torch.zeros([batch_size, relations.shape[1], self.encoding_size]).to(device)

        # obtain relation encodings
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            rel_encoding_chunk = self._encode_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_encoding[:, i:i + self._max_pairs, :] = rel_encoding_chunk

        rel_clf = self._classify_relations(rel_encoding)

        return entity_clf, rel_clf

    def _forward_eval(self, entity_knn_module, rel_knn_module, entity_entries: List[List[Dict]], 
                      encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor, 
                      entity_sizes: torch.tensor, entity_spans: torch.tensor = None, 
                      entity_sample_mask: torch.tensor = None, verbose=True):
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        # enhance hidden features
        orig_shape = h.shape
        h = self.feature_enhancer.prepare_input(h, context_mask)
        h = self.feature_enhancer(h)
        h = self.feature_enhancer.prepare_output(h, orig_shape)

        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]
        ctx_size = context_mask.shape[-1]
        device = self.entity_encoder.weight.device

        # encode and classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_encoding, entity_spans_pool = self._encode_entities(encodings, h, entity_masks, size_embeddings)
        entity_encoding_reshaped = entity_encoding.view(entity_encoding.shape[0]*entity_encoding.shape[1], -1).cpu()
        entity_types, entity_neighbors = entity_knn_module.infer_(entity_encoding_reshaped, int, self._type_key)
        
        # print neighbor entities
        if verbose:
            print('*'*50)
            print("entity neighbors:")
            entity_entries_flat = []
            for entry in entity_entries: 
                entity_entries_flat += entry
            for i, neighbors in enumerate(entity_neighbors):
                if entity_types[i] == 0: 
                    continue
                print("[ENT] {} >> {}".format(entity_entries_flat[i]["phrase"], entity_types[i]))
                for j in range(min(len(neighbors), 5)):
                    n = neighbors[j]
                    print("\t", n["phrase"], n["type"], n["type_index"])
                print()

        entity_types = torch.tensor(entity_types).view(entity_encoding.shape[0], entity_encoding.shape[1]).to(device)
        entity_clf = torch.zeros([entity_encoding.shape[0], entity_encoding.shape[1],
                                  self._entity_types], dtype=torch.long).to(device)
        entity_clf.scatter_(2, entity_types.unsqueeze(2), 1)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks, rel_entries = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_mask, entity_entries, ctx_size)
        rel_masks = rel_masks.float()
        rel_sample_masks = rel_sample_masks.float()
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_encoding = torch.zeros([batch_size, relations.shape[1], self.encoding_size]).to(device)

        # obtain relation encodings
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            rel_encoding_chunk = self._encode_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_encoding[:, i:i + self._max_pairs, :] = rel_encoding_chunk
        rel_encoding_reshaped = rel_encoding.view(rel_encoding.shape[0]*rel_encoding.shape[1], -1).cpu()
        
        # encode and classify relations
        rel_types, rel_neighbors = rel_knn_module.infer_(rel_encoding_reshaped, int, self._type_key)

        # print neighbor relations
        if verbose:
            print('*'*50)
            rel_entries_flat = []
            for entry in rel_entries: 
                rel_entries_flat += entry
            for i, neighbors in enumerate(rel_neighbors):
                if rel_types[i] == 0: 
                    continue
                print("[REL] {} >> {}".format(rel_entries_flat[i]["phrase"], rel_types[i]))
                for j in range(min(len(neighbors), 5)):
                    n = neighbors[j]
                    print("\t", n["phrase"], n["type"], n["type_index"])
                print()

        rel_types = torch.LongTensor(rel_types).view(rel_encoding.shape[0], rel_encoding.shape[1]).to(device)
        rel_clf = torch.zeros([rel_encoding.shape[0], rel_encoding.shape[1],
                               self._relation_types], dtype=torch.float32).to(device)

        rel_clf.scatter_(2, rel_types.unsqueeze(2), 1)
        rel_clf = rel_clf[:, :, 1:] # exclude 'none' prediction for multi-label prediction
        
        rel_clf = rel_clf * rel_sample_masks  # mask

        return entity_clf, rel_clf, relations


    def _forward_encode(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor,
                        entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]
        device = self.entity_encoder.weight.device

        # encode and classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_encoding, entity_spans_pool = self._encode_entities(encodings, h, entity_masks, size_embeddings)

        # prepare relation encoding
        rel_masks = rel_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_encoding = torch.zeros([batch_size, relations.shape[1], self.encoding_size]).to(device)

        # obtain relation encodings
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            rel_encoding_chunk = self._encode_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_encoding[:, i:i + self._max_pairs, :] = rel_encoding_chunk

        return entity_encoding, rel_encoding

    def _classify_entities(self, entity_encoding, verification=False):
        # cosine similarities of every possible entity encoding pair in the batch
        cosine_similarities = torch.einsum('abc, ijc -> abij', entity_encoding, entity_encoding)

        # einsum verification (at least each element has similarity 1 with itself)
        if verification:
            with torch.no_grad():
                is_close_bools = cosine_similarities.isclose(torch.tensor([1.00], device=self.entity_encoder.weight.device))
                is_close_sum = is_close_bools.int().sum().item()
                assert(is_close_sum >= entity_encoding.shape[0]*entity_encoding.shape[1])

        # normalize cosine similarity from [-1, 1] to [0, 1], and clip float precision errors
        normalized_similarities = (cosine_similarities + 1) / 2
        normalized_similarities = normalized_similarities.clamp(0, 1)

        return normalized_similarities

    def _encode_entities(self, encodings, h, entity_masks, size_embeddings):
        # max pool entity candidate spans
        entity_spans_pool = entity_masks.unsqueeze(-1) * h.unsqueeze(1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # encode entity candidates
        entity_encoding = self.entity_encoder(entity_repr)

        # normalize encoding to unit length for cosine similarity
        entity_encoding = f.normalize(entity_encoding, dim=2, p=2)

        return entity_encoding, entity_spans_pool

    def _classify_relations(self, rel_encoding, verification=False):
        # cosine similarity of every possible relation encoding pair in the batch
        cosine_similarities = torch.einsum('abc, ijc -> abij', rel_encoding, rel_encoding)

        # einsum verification (at least each element has similarity 1 with itself)
        if verification:
            with torch.no_grad():
                is_close_bools = cosine_similarities.isclose(torch.tensor([1.00], device=self.rel_encoder.weight.device))
                is_close_sum = is_close_bools.int().sum().item()
                assert(is_close_sum >= rel_encoding.shape[0]*rel_encoding.shape[1])

        # normalize cosine similarity from [-1, 1] to [0, 1], and clip float precision errors
        normalized_similarities = (cosine_similarities + 1) / 2
        normalized_similarities = normalized_similarities.clamp(0, 1)

        return normalized_similarities


    def _encode_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        rel_ctx = rel_masks * h
        rel_ctx = rel_ctx.max(dim=2)[0]

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # encode relation candidates
        rel_encoding = self.rel_encoder(rel_repr)

        # normalize encoding to unit length for cosine similarity
        rel_encoding = f.normalize(rel_encoding, dim=2, p=2)

        return rel_encoding


    #TODO: Needs checking of relation entries
    def _filter_spans(self, entity_clf, entity_spans, entity_sample_mask, entity_entries, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_mask.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []
        batch_rel_entries = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []
            rel_entries = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_entries = [entity_entries[i][j] for j in non_zero_indices] 
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for n, (i1, s1) in enumerate(zip(non_zero_indices, non_zero_spans)):
                for m, (i2, s2) in enumerate(zip(non_zero_indices, non_zero_spans)):
                    if i1 != i2:
                        rels.append((i1, i2))
                        phrase = "|{}| <TBD> |{}|".format(non_zero_entries[n]["phrase"], non_zero_entries[m]["phrase"])
                        rel_entries.append({"phrase": phrase, "type": "<TBD>", self._type_key: "<TBD>"})
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
                phrase = ""
                batch_rel_entries.append([{"phrase": phrase, "type": "<TBD>", self._type_key: "<TBD>"}])
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))
                batch_rel_entries.append(rel_entries)

        # stack
        device = self.rel_encoder.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device).unsqueeze(-1)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device).unsqueeze(-1)
        batch_rel_entries = util.padded_entries(batch_rel_entries)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks, batch_rel_entries


    def forward(self, *args, mode="train", **kwargs):
        f_forward = {
            "train": self._forward_train,
            "eval": self._forward_eval,
            "encode": self._forward_encode
        }.get(mode)
        return f_forward(*args, **kwargs)


_MODELS = {
    'spert': SpERT,
    'speer': SpEER
}


def get_model(name):
    return _MODELS[name]
