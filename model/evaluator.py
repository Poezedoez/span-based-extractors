import os
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from model.entities import Document, Dataset, EntityType
from model.input_reader import JsonInputReader
from model.opt import jinja2
from model.sampling import EvalTensorBatch

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, example_count: int, example_path: str,
                 epoch: int, dataset_label: str, entities_only=False, relations_only=False):
        print("Initializing evaluator... for dataset {}".format(dataset_label))
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold
        self._entities_only = entities_only
        self._relations_only = relations_only

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._example_count = example_count
        self._examples_path = example_path

        self._sequences = []
        self._tok2orig = []

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._predictions_json = []
        self._evaluated_sequences = 0

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation

        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                   batch_rels: torch.tensor, batch: EvalTensorBatch, return_conversions=False):
        batch_size = batch_rel_clf.shape[0]
        rel_class_count = batch_rel_clf.shape[2]

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= batch.entity_sample_masks.long()

        batch_rel_clf = batch_rel_clf.view(batch_size, -1)

        # apply threshold to relations
        if self._rel_filter_threshold > 0:
            batch_rel_clf[batch_rel_clf < self._rel_filter_threshold] = 0

        batch_entities = []
        batch_relations = []
        for i in range(batch_size):
            # get model predictions for sample
            rel_clf = batch_rel_clf[i]
            entity_types = batch_entity_types[i]

            # get predicted relation labels and corresponding entity pairs
            rel_nonzero = rel_clf.nonzero().view(-1)
            rel_scores = rel_clf[rel_nonzero]

            # rel_types = (rel_nonzero % rel_class_count)
            rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1)
            rel_indices = rel_nonzero // rel_class_count

            rels = batch_rels[i][rel_indices]

            # get masks of entities in relation
            rel_entity_spans = batch.entity_spans[i][rels].long()

            # get predicted entity types
            rel_entity_types = torch.zeros([rels.shape[0], 2])
            if rels.shape[0] != 0:
                rel_entity_types = torch.stack([entity_types[rels[j]] for j in range(rels.shape[0])])
            
            # get original tokens
            tokens = self._sequences[self._evaluated_sequences]

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_types.nonzero().view(-1)
            valid_entity_types = entity_types[valid_entity_indices]
            valid_entity_spans = batch.entity_spans[i][valid_entity_indices]
            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)

            sample_pred_entities, mapping, json_entities = self._convert_pred_entities(valid_entity_types, valid_entity_spans,
                                                               valid_entity_scores)
            self._pred_entities.append(sample_pred_entities)

            # convert predicted relations for evaluation
            sample_pred_relations, json_relations = self._convert_pred_relations(rel_types, rel_entity_spans,
                                                                 rel_entity_types, rel_scores, mapping)
            self._pred_relations.append(sample_pred_relations)

            batch_entities.append(sample_pred_entities)
            batch_relations.append(sample_pred_relations)

            self._predictions_json.append({"tokens": tokens, 
                                           "entities": json_entities, 
                                           "relations": json_relations, 
                                           "id": hash(" ".join(tokens))})
                                           
            self._evaluated_sequences += 1
            

        return batch_entities, batch_relations


    def eval_batch_entities(self, batch_entity_clf: torch.tensor, 
                            batch: EvalTensorBatch, return_conversions=False):
        batch_size = batch_entity_clf.shape[0]

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= batch.entity_sample_masks.long()

        batch_entities = []
        for i in range(batch_size):
            # get model predictions for sample
            entity_types = batch_entity_types[i]

            
            # get original tokens
            tokens = self._sequences[self._evaluated_sequences]

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_types.nonzero().view(-1)
            valid_entity_types = entity_types[valid_entity_indices]
            valid_entity_spans = batch.entity_spans[i][valid_entity_indices]
            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)

            sample_pred_entities, mapping, json_entities = self._convert_pred_entities(valid_entity_types, valid_entity_spans,
                                                               valid_entity_scores)
            self._pred_entities.append(sample_pred_entities)

            batch_entities.append(sample_pred_entities)

            self._predictions_json.append({"tokens": tokens, 
                                           "entities": json_entities, 
                                           "relations": [], 
                                           "id": hash(" ".join(tokens))})
                                           
            self._evaluated_sequences += 1
            
        return batch_entities

        
    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the spans of the two "
              "related entities are predicted correctly (entity type is not considered)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=False)
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the two "
              "related entities are predicted correctly (in span and entity type)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=True)
        rel_nec_eval = self._score(gt, pred, print_results=True)

        return ner_eval, rel_eval, rel_nec_eval

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        entity_examples = []
        rel_examples = []
        rel_examples_nec = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            # relations
            # without entity types
            rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_entity_types=False, to_html=self._rel_to_html)
            rel_examples.append(rel_example)

            # with entity types
            rel_example_nec = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                    include_entity_types=True, to_html=self._rel_to_html)
            rel_examples_nec.append(rel_example_nec)

        label, epoch = self._dataset_label, self._epoch

        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % ('entities', label, epoch),
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('entities_sorted', label, epoch),
                             template='entity_examples.html')

        # relations
        # without entity types
        self._store_examples(rel_examples[:self._example_count],
                             file_path=self._examples_path % ('rel', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_sorted', label, epoch),
                             template='relation_examples.html')

        # with entity types
        self._store_examples(rel_examples_nec[:self._example_count],
                             file_path=self._examples_path % ('rel_nec', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples_nec[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_nec_sorted', label, epoch),
                             template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            tokens = doc.tokens
            gt_relations = doc.relations
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]
            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
            self._sequences.append([t.phrase for t in tokens._tokens])
            self._tok2orig.append(self._map_tokens(tokens._tokens))
            self._gt_relations.append(sample_gt_relations)
            self._gt_entities.append(sample_gt_entities)
    
    def _map_tokens(self, tokens):
        tok2orig = [0] # special token
        for i, t in enumerate(tokens): # skip special tokens
            start, end = t.span_start, t.span_end
            for _ in range(start, end):
                tok2orig.append(i)
            prev = start
        tok2orig.append(len(tokens)) # special token

        return tok2orig

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []
        json_entities = []
        position_mapping = {}
        
        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)
            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()
            converted_pred = (start, end, entity_type, score)
            token_range = self._tok2orig[self._evaluated_sequences][start:end+1]
            orig_start, orig_end = token_range[0], token_range[-1]
            position_mapping[(orig_start, orig_end)] = len(converted_preds)
            json_entities.append({"start": orig_start, "end": orig_end, "type": entity_type.short_name})
            converted_preds.append(converted_pred)

        return converted_preds, position_mapping, json_entities

    def _convert_pred_relations(self, pred_rel_types: torch.tensor, pred_entity_spans: torch.tensor,
                                pred_entity_types: torch.tensor, pred_scores: torch.tensor,
                                position_mapping: Dict[Tuple, int]):
        converted_rels = []
        json_relations = []
        check = set()

        for i in range(pred_rel_types.shape[0]):
            label_idx = pred_rel_types[i].item()
            pred_rel_type = self._input_reader.get_relation_type(label_idx)
            pred_head_type_idx, pred_tail_type_idx = pred_entity_types[i][0].item(), pred_entity_types[i][1].item()
            pred_head_type = self._input_reader.get_entity_type(pred_head_type_idx)
            pred_tail_type = self._input_reader.get_entity_type(pred_tail_type_idx)
            score = pred_scores[i].item()

            spans = pred_entity_spans[i]
            head_start, head_end = spans[0].tolist()
            tail_start, tail_end = spans[1].tolist()

            rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type)
            converted_rel = self._adjust_rel(rel)
            head_start, head_end = converted_rel[0][0], converted_rel[0][1]
            tail_start, tail_end = converted_rel[1][0], converted_rel[1][1]
            t2o = self._tok2orig[self._evaluated_sequences]
            head_range = t2o[head_start:head_end+1]
            tail_range = t2o[tail_start:tail_end+1]
            head_start, head_end = head_range[0], head_range[-1]
            tail_start, tail_end = tail_range[0], tail_range[-1]
            final_head = position_mapping[(head_start, head_end)]
            final_tail = position_mapping[(tail_start, tail_end)]
            json_relations.append({"head": final_head, "tail": final_tail, "type": pred_rel_type.short_name})

            if converted_rel not in check:
                check.add(converted_rel)
                converted_rels.append(tuple(list(converted_rel) + [score]))

        return converted_rels, json_relations

    def _adjust_rel(self, rel: Tuple):
        adjusted_rel = rel
        if rel[-1].symmetric:
            head, tail = rel[:2]
            if tail[0] < head[0]:
                adjusted_rel = tail, head, rel[-1]

        return adjusted_rel

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        encoding = doc.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
