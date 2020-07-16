import argparse
import math
import os
import datetime
import logging
import time
import sys

from nearest_neighbert import NearestNeighBERT as NN
import torch
from tqdm import tqdm
from torch.nn import DataParallel
from torch.optim import Optimizer
import torch.nn.functional as F
import transformers
from transformers import AdamW
from transformers import BertTokenizer
from transformers import PreTrainedModel

from model import util
from model.opt import tensorboardX
from model import models
from model.entities import Dataset
from model.evaluator import Evaluator
from model.input_reader import JsonInputReader, BaseInputReader
from model.loss import Loss, SpETLoss, SpERTLoss, SpEERLoss, SpRTLoss
from model.sampling import Sampler

from typing import List, Dict, Tuple, Any, Type

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class BaseTrainer:
    """ Trainer base class with common methods """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._debug = self.args.debug

        # logging
        name = str(datetime.datetime.now()).replace(' ', '_')
        self.args.label = self.args.label if self.args.label else ''
        self._log_path = os.path.join(self.args.log_path, self.args.label, name) if not (
            args.timestamp_given) else self.args.log_path
        util.create_directories_dir(self._log_path)

        if hasattr(args, 'save_path'):
            save_path = os.path.join(self.args.save_path, self.args.label, name) if not (
                args.timestamp_given) else self.args.save_path
            self._save_path = save_path
            util.create_directories_dir(self._save_path)

        self._log_paths = dict()

        # file + console logging
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self._logger = logging.getLogger()
        util.reset_logger(self._logger)

        file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))
        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)

        if self._debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)

        # tensorboard summary
        self._summary_writer = tensorboardX.SummaryWriter(self._log_path) if tensorboardX is not None else None

        self._best_results = dict()
        self._log_arguments()

        # CUDA devices
        self._device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self._gpu_count = torch.cuda.device_count()

        # set seed
        if args.seed is not None:
            util.set_seed(args.seed)

    def _add_dataset_logging(self, *labels, data: Dict[str, List[str]]):
        for label in labels:
            dic = dict()

            for key, columns in data.items():
                path = os.path.join(self._log_path, '%s_%s.csv' % (key, label))
                util.create_csv(path, *columns)
                dic[key] = path

            self._log_paths[label] = dic
            self._best_results[label] = 0

    def _log_arguments(self):
        util.save_dict(self._log_path, self.args, 'args')
        if self._summary_writer is not None:
            util.summarize_dict(self._summary_writer, self.args, 'args')

    def _log_tensorboard(self, dataset_label: str, data_label: str, data: object, iteration: int):
        if self._summary_writer is not None:
            self._summary_writer.add_scalar('data/%s/%s' % (dataset_label, data_label), data, iteration)

    def _log_csv(self, dataset_label: str, data_label: str, *data: Tuple[object]):
        logs = self._log_paths[dataset_label]
        util.append_csv(logs[data_label], *data)

    def _save_best(self, model, optimizer, accuracy, iteration, label, extra=None):
        if accuracy > self._best_results[label]:
            self._logger.info("[%s] Best model in iteration %s: %s%% accuracy" % (label, iteration, accuracy))
            self._save_model(self._save_path, model, iteration,
                             optimizer=optimizer if self.args.save_optimizer else None,
                             save_as_best=True, name='model_%s' % label, extra=extra)
            self._best_results[label] = accuracy

    def _save_model(self, save_path: str, model: PreTrainedModel, iteration: int, optimizer: Optimizer = None,
                    save_as_best: bool = False, extra: dict = None, include_iteration: int = True, name: str = 'model'):
        extra_state = dict(iteration=iteration)

        if optimizer:
            extra_state['optimizer'] = optimizer.state_dict()

        if extra:
            extra_state.update(extra)

        if save_as_best:
            dir_path = os.path.join(save_path, '%s_best' % name)
        else:
            dir_name = '%s_%s' % (name, iteration) if include_iteration else name
            dir_path = os.path.join(save_path, dir_name)

        util.create_directories_dir(dir_path)

        if isinstance(model, DataParallel):
            model.module.save_pretrained(dir_path)
        else:
            model.save_pretrained(dir_path)
        state_path = os.path.join(dir_path, 'extra.state')
        torch.save(extra_state, state_path)

    def _get_lr(self, optimizer):
        lrs = []
        for group in optimizer.param_groups:
            lr_scheduled = group['lr']
            lrs.append(lr_scheduled)
        return lrs


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        # tokenizer_path = os.path.join(args.tokenizer_path, 'vocab.txt')
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        # sampler (create and batch training/evaluation samples)
        self._sampler = Sampler(processes=args.sampling_processes, limit=args.sampling_limit)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader, 
              skip_saving=False):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # parallelize model
        if self._device.type != 'cpu':
            model = torch.nn.DataParallel(model)
            print("GPU's available: ", torch.cuda.device_count())
        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        if args.skip_relations:
            compute_loss = SpETLoss(entity_criterion, model, optimizer, scheduler, args.max_grad_norm)
        else:
            compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch,
                              input_reader.context_size, input_reader.relation_type_count)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

        # save final model
        name = 'final_model' if not args.timestamp_given else ''
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        if not args.skip_saving:
            self._save_model(self._save_path, model, global_iteration,
                            optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                            include_iteration=False, name=name)
            self._logger.info("Saved in: %s" % self._save_path)
        self._logger.info("Logged in: %s" % self._log_path)
        

        self._sampler.join()

    def eval(self, eval_path: str, train_path: str, predicted_entities_path: str, 
             types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % eval_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read({dataset_label: eval_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # additional model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)

        model.to(self._device)

        # evaluate
        ner_eval, rel_eval, raw_output = self._eval(model, input_reader.get_dataset(dataset_label), input_reader)
        self._logger.info("Logged in: %s" % self._log_path)

        self._sampler.join()

        # TODO: fix this function
        # util.convert_to_json_dataset(raw_output, self._log_path, save=True)

        return ner_eval, rel_eval

    def infer(self, document_data: Dict[str, Any], types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = document_data['guid']

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read(document_data)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # additional model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)
        model.to(self._device)

        # do inference
        _, _, outputs = self._eval(model, input_reader.get_dataset(dataset_label), input_reader, inference_only=True)

        return outputs

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, context_size: int, rel_type_count: int):
        self._logger.info("Train epoch: %s" % epoch)

        # randomly shuffle data
        order = torch.randperm(dataset.document_count)
        sampler = self._sampler.create_train_sampler(dataset, self.args.train_batch_size, self.args.max_span_size,
                                                     context_size, self.args.neg_entity_count,
                                                     self.args.neg_relation_count, order=order, truncate=True)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(sampler, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = batch.to(self._device)

            # relation types to one-hot encoding
            rel_types_onehot = torch.zeros([batch.rel_types.shape[0], batch.rel_types.shape[1],
                                            rel_type_count], dtype=torch.float32).to(self._device)
            rel_types_onehot.scatter_(2, batch.rel_types.unsqueeze(2), 1)
            rel_types_onehot = rel_types_onehot[:, :, 1:]  # all zeros for 'none' relation

            # forward step
            entity_logits, rel_logits = model(batch.encodings, batch.ctx_masks, batch.entity_masks,
                                              batch.entity_sizes, batch.rels, batch.rel_masks)

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(rel_logits, rel_types_onehot, entity_logits,
                                              batch.entity_types, batch.rel_sample_masks, batch.entity_sample_masks)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration


    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0, inference_only: bool = False):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module
        
        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.example_count,
                              self._examples_path, epoch, dataset.label)

        # create batch sampler
        sampler = self._sampler.create_eval_sampler(dataset, self.args.eval_batch_size, self.args.max_span_size,
                                                    input_reader.context_size, truncate=False)
        sequences = []
        entities = []
        relations = []
        with torch.no_grad():
            model.eval()
            print("evaluating...")
            print("DEVICE:", self._device)
            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            batch_count = 0
            for batch in tqdm(sampler, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = batch.to(self._device)
                batch_size = batch.encodings.shape[0]
                # run model (forward pass)
                entity_clf, rel_clf, rels = model(batch.encodings, batch.ctx_masks, batch.entity_masks,
                                                  batch.entity_sizes, batch.entity_spans, batch.entity_sample_masks,
                                                  evaluate=True)

                # evaluate batch
                # get maximum activation (index of predicted entity type)
                batch_entity_types = entity_clf.argmax(dim=-1)
                # apply entity sample mask
                batch_entity_types *= batch.entity_sample_masks.long()

                batch_entities, batch_relations = evaluator.eval_batch(entity_clf, 
                    rel_clf, rels, batch, return_conversions=True)
                for i in range(batch_size):
                    sequences.append(evaluator._sequences[batch_count*batch_size+i])
                    entities.append(batch_entities[i])
                    relations.append(batch_relations[i])
                batch_count += 1

        if not inference_only:
            global_iteration = epoch * updates_epoch + iteration
            ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
            self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                        epoch, iteration, global_iteration, dataset.label)

            if self.args.store_examples:
                evaluator.store_examples()

        return ner_eval, rel_eval, (sequences, entities, relations)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

class SpEERTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        # tokenizer_path = os.path.join(args.tokenizer_path, 'vocab.txt')
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        # sampler (create and batch training/evaluation samples)
        self._sampler = Sampler(processes=args.sampling_processes, limit=args.sampling_limit)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader,
              skip_saving=False):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        # temp = args.model_path if args.model_path else "NONOnonnno"
        # print(temp)
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            encoding_size=args.encoding_size,
                                            feature_enhancer=args.feature_enhancer)

        # SpERT/SpEER is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # parallelize model
        if self._device.type != 'cpu':
            model = torch.nn.DataParallel(model)
            print("GPU's available: ", torch.cuda.device_count())
        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)

        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCELoss(reduction='none')
        entity_criterion = torch.nn.BCELoss(reduction='none')
        compute_loss = SpEERLoss(entity_criterion, rel_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            entity_knn_module, rel_knn_module = self._index(model, train_dataset, input_reader, args.k)
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch,
                              input_reader.context_size, input_reader.relation_type_count, input_reader.entity_type_count)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                entity_knn_module, rel_knn_module = self._index(model, train_dataset, input_reader, args.k)
                self._eval(model, validation_dataset, entity_knn_module, rel_knn_module, 
                           input_reader, epoch + 1, updates_epoch)

        # save final model
        name = 'final_model' if not args.timestamp_given else ''
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        if not args.skip_saving:
            self._save_model(self._save_path, model, global_iteration,
                            optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                            include_iteration=False, name=name)
            self._logger.info("Saved in: %s" % self._save_path)

        self._logger.info("Logged in: %s" % self._log_path)
        self._sampler.join()

        return model

    def eval(self, eval_path: str, train_path: str, predicted_entities_path: str, 
             types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label = 'train'
        eval_label = 'test'

        # self._init_train_logging(train_label)
        self._init_eval_logging(eval_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)

        input_reader.read({train_label: train_path})
        train_dataset = input_reader.get_dataset(train_label)
        self._log_datasets(input_reader)

        input_reader.read({eval_label: eval_path})
        eval_dataset = input_reader.get_dataset(eval_label)
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # additional model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            encoding_size=args.encoding_size,
                                            feature_enhancer=self.args.feature_enhancer)

        model.to(self._device)

        # evaluate
        entity_knn_module, rel_knn_module = self._index(model, train_dataset, input_reader, self.args.k)
        ner_eval, rel_eval, raw_output = self._eval(model, eval_dataset, entity_knn_module, rel_knn_module, input_reader)
        self._logger.info("Logged in: %s" % self._log_path)

        self._sampler.join()

        # TODO: fix this function
        # util.convert_to_json_dataset(raw_output, self._log_path, save=True)

        return ner_eval, rel_eval

    def infer(self, document_data: Dict[str, Any], types_path: str, 
              input_reader_cls: BaseInputReader, entity_knn_module: Any, rel_knn_module: Any):
        args = self.args
        dataset_label = document_data['guid']

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read(document_data)
        inference_dataset = input_reader.get_dataset(dataset_label)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # additional model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)

        model.to(self._device)

        #TODO: this needs to be moved to load inference model    
        # entity_knn_module, rel_knn_module = self._index(model, train_dataset, input_reader)

        # do inference
        _, _, raw_output = self._eval(model, inference_dataset, entity_knn_module, rel_knn_module, input_reader)

        return outputs

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, context_size: int, rel_type_count: int, entity_type_count: int):
        self._logger.info("Train epoch: %s" % epoch)

        # randomly shuffle data
        order = torch.randperm(dataset.document_count)
        sampler = self._sampler.create_train_sampler(dataset, self.args.train_batch_size, self.args.max_span_size,
                                                     context_size, self.args.neg_entity_count,
                                                     self.args.neg_relation_count, order=order, truncate=True)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        epoch_loss = 0
        for batch in tqdm(sampler, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = batch.to(self._device)

            # entity types to one-hot encoding
            entity_types_onehot = torch.zeros([batch.entity_types.shape[0], batch.entity_types.shape[1],
                                            entity_type_count], dtype=torch.long).to(self._device)
            entity_types_onehot.scatter_(2, batch.entity_types.unsqueeze(2), 1)

            # relation types to one-hot encoding
            rel_types_onehot = torch.zeros([batch.rel_types.shape[0], batch.rel_types.shape[1],
                                            rel_type_count], dtype=torch.long).to(self._device)
            rel_types_onehot.scatter_(2, batch.rel_types.unsqueeze(2), 1)

            # forward step
            entity_logits, rel_logits = model(batch.encodings, batch.ctx_masks, batch.entity_masks,
                                              batch.entity_sizes, batch.rels, batch.rel_masks, 
                                              mode="train")

            # create BCE entity types (TODO: find vectorized way? scatter?)
            s_batch, s_spans, _ = entity_types_onehot.shape
            entity_types_bce = torch.zeros(s_batch, s_spans, s_batch, s_spans, device=self._device)
            for i in range(0, s_batch):
                for j in range(0, s_spans):
                    type_ = batch.entity_types[i, j]
                    entity_types_bce[i, j] = entity_types_onehot[:, :, type_]

            # create BCE rel types (TODO: find vectorized way? scatter?)
            s_batch, s_spans, _ = rel_types_onehot.shape
            rel_types_bce = torch.zeros(s_batch, s_spans, s_batch, s_spans, device=self._device)
            for i in range(0, s_batch):
                for j in range(0, s_spans):
                    type_ = batch.rel_types[i, j]
                    rel_types_bce[i, j, :, :] = rel_types_onehot[:, :, type_]

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits, entity_types_bce, batch.entity_sample_masks, 
                                              rel_logits, rel_types_bce, batch.rel_sample_masks)
            epoch_loss += batch_loss

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        print("Loss epoch {}: {}".format(epoch, epoch_loss/self.args.train_batch_size))

        return iteration


    def _index(self, model: torch.nn.Module, dataset: Dataset,
               input_reader: JsonInputReader, k: int, epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Index: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # load knn modules (with hardcoded settings)
        entity_knn_module = NN.NearestNeighBERT(k=k, f_similarity="IP", indicator="entity_knn_module", device="cpu")
        rel_knn_module = NN.NearestNeighBERT(k=k, f_similarity="IP", indicator="rel_knn_module", device="cpu")
        entity_knn_module.ready_training(self.args.tokenizer_path, self.args.encoding_size)
        rel_knn_module.ready_training(self.args.tokenizer_path, self.args.encoding_size)

        train_sampler = self._sampler.create_train_sampler(dataset, self.args.eval_batch_size, self.args.max_span_size,
                                                    input_reader.context_size, self.args.neg_entity_count,
                                                    self.args.neg_relation_count, truncate=False, type_key=self.args.type_key)
        # encode dataset
        with torch.no_grad():
            print("Encoding train examples for evaluation...")
            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(train_sampler, total=total):
                batch = batch.to(self._device)
                # forward encode
                entity_encoding, rel_encoding = model(batch.encodings, batch.ctx_masks, batch.entity_masks, 
                                                        batch.entity_sizes, batch.rels, batch.rel_masks,
                                                        mode="encode")
                # flatten encodings and entries
                # entity_encoding = entity_encoding.view(entity_encoding.shape[0]*entity_encoding.shape[1], -1).cpu() 
                entity_entries_reshaped = []
                entity_encodings_reshaped = []
                for i, entries in enumerate(batch.entity_entries):
                    for j, entry in enumerate(entries):
                        if entry["type_string"] != "<PAD>":
                            entity_entries_reshaped.append(entry)
                            entity_encodings_reshaped.append(entity_encoding[i][j].tolist())
                
                # rel_encoding = rel_encoding.view(rel_encoding.shape[0]*rel_encoding.shape[1], -1).cpu()
                rel_entries_reshaped = []
                rel_encodings_reshaped = []
                for i, entries in enumerate(batch.rel_entries):
                    for j, entry in enumerate(entries):
                        if entry["type_string"] != "<PAD>":
                            rel_entries_reshaped.append(entry)
                            rel_encodings_reshaped.append(rel_encoding[i][j].tolist())
                
                # index encodings
                assert(len(entity_encodings_reshaped)==len(entity_entries_reshaped))
                assert(len(rel_encodings_reshaped)==len(rel_entries_reshaped))
                entity_knn_module.train_(entity_encodings_reshaped, entity_entries_reshaped)
                if rel_encodings_reshaped: # in case no relations
                    rel_knn_module.train_(rel_encodings_reshaped, rel_entries_reshaped)

        return entity_knn_module, rel_knn_module
            

    def _eval(self, model: torch.nn.Module, dataset: Dataset, entity_knn_module: Any,
              rel_knn_module: Any, input_reader: JsonInputReader, 
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0, inference_only=False):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.example_count,
                              self._examples_path, epoch, dataset.label)

        # create batch sampler
        sampler = self._sampler.create_eval_sampler(dataset, self.args.eval_batch_size, self.args.max_span_size,
                                                    input_reader.context_size, truncate=False, type_key=self.args.type_key)         
        sequences = []
        entities = []
        relations = []
        with torch.no_grad():
            model.eval()
            batch_count = 0
            print("evaluating...")
            print("DEVICE:", self._device)
            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)

            for batch in tqdm(sampler, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = batch.to(self._device)
                batch_size = batch.encodings.shape[0]
                # run model (forward pass)
                entity_clf, rel_clf, rels = model(entity_knn_module, rel_knn_module, batch.entity_entries, self.args.type_key,
                                            batch.encodings, batch.ctx_masks, batch.entity_masks,batch.entity_sizes, 
                                            batch.entity_spans, batch.entity_sample_masks, mode="eval")

                
                # evaluate batch (TODO: what is this?) This block necessary?
                # get maximum activation (index of predicted entity type)
                batch_entity_types = entity_clf.argmax(dim=-1)
                # apply entity sample mask 
                batch_entity_types *= batch.entity_sample_masks.long()
                ############

                batch_entities, batch_relations = evaluator.eval_batch(entity_clf, rel_clf, rels, batch, return_conversions=True)
                for i in range(batch_size):
                    sequences.append(evaluator._sequences[batch_count*batch_size+i])
                    entities.append(batch_entities[i])
                    relations.append(batch_relations[i])
                batch_count += 1

        if not inference_only:
            global_iteration = epoch * updates_epoch + iteration
            ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
            self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                        epoch, iteration, global_iteration, dataset.label)

            if self.args.store_examples:
                evaluator.store_examples() 

        return ner_eval, rel_eval, (sequences, entities, relations)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

class SpRTTrainer(BaseTrainer):
    """ Relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        # tokenizer_path = os.path.join(args.tokenizer_path, 'vocab.txt')
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        # sampler (create and batch training/evaluation samples)
        self._sampler = Sampler(processes=args.sampling_processes, limit=args.sampling_limit)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader, skip_saving=False):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # parallelize model
        if self._device.type != 'cpu':
            model = torch.nn.DataParallel(model)
            print("GPU's available: ", torch.cuda.device_count())
        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        compute_loss = SpRTLoss(rel_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch,
                              input_reader.context_size, input_reader.relation_type_count)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

        # save final model
        name = 'final_model' if not args.timestamp_given else ''
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        if not args.skip_saving:
            self._save_model(self._save_path, model, global_iteration,
                            optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                            include_iteration=False, name=name)
            self._logger.info("Saved in: %s" % self._save_path)

        self._logger.info("Logged in: %s" % self._log_path)
        self._sampler.join()

        return model

    def eval(self, eval_path: str, train_path: str, predicted_entities_path: str, 
             types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        predicted_entities_label = 'predicted_entities'
        eval_label = 'test'

        self._init_eval_logging(eval_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)

        input_reader.read({predicted_entities_label: predicted_entities_path})
        predicted_entities_dataset = input_reader.get_dataset(predicted_entities_label)
        self._log_datasets(input_reader)

        input_reader.read({eval_label: eval_path})
        eval_dataset = input_reader.get_dataset(eval_label)
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # additional model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)

        model.to(self._device)

        # evaluate
        ner_eval, rel_eval, raw_output = self._eval(model, eval_dataset, predicted_entities_dataset, input_reader)
        self._logger.info("Logged in: %s" % self._log_path)
        self._sampler.join()

        # TODO: fix this function
        # util.convert_to_json_dataset(raw_output, self._log_path, save=True)

        return ner_eval, rel_eval

    def infer(self, document_data: Dict[str, Any], types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = document_data['guid']

        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read(document_data)
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            # additional model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            # no node for 'none' class
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            feature_enhancer=self.args.feature_enhancer)

        model.to(self._device)

        # do inference
        _, _, outputs = self._eval(model, input_reader.get_dataset(dataset_label), input_reader, inference_only=True)

        return outputs

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, context_size: int, rel_type_count: int):
        self._logger.info("Train epoch: %s" % epoch)

        # randomly shuffle data
        order = torch.randperm(dataset.document_count)
        sampler = self._sampler.create_train_sampler(dataset, self.args.train_batch_size, self.args.max_span_size,
                                                     context_size, self.args.neg_entity_count,
                                                     self.args.neg_relation_count, order=order, truncate=True, relations_only=True)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(sampler, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = batch.to(self._device)

            # relation types to one-hot encoding
            rel_types_onehot = torch.zeros([batch.rel_types.shape[0], batch.rel_types.shape[1],
                                            rel_type_count], dtype=torch.float32).to(self._device)
            rel_types_onehot.scatter_(2, batch.rel_types.unsqueeze(2), 1)
            rel_types_onehot = rel_types_onehot[:, :, 1:]  # all zeros for 'none' relation

            # forward step
            rel_logits = model(batch.encodings, batch.ctx_masks, batch.entity_masks,
                                              batch.entity_sizes, batch.rels, batch.rel_masks)

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(rel_logits, rel_types_onehot, batch.rel_sample_masks)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, eval_dataset: Dataset, predicted_entities_dataset: Dataset, 
              input_reader: JsonInputReader, epoch: int = 0, updates_epoch: int = 0, 
              iteration: int = 0, inference_only: bool = False):
        self._logger.info("Evaluate: %s" % eval_dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        assert len(eval_dataset.document_count)==len(predicted_entities_dataset.document_count)
        
        # create evaluator
        evaluator = Evaluator(eval_dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.example_count,
                              self._examples_path, epoch, eval_dataset.label)

        # create batch sampler
        sampler = self._sampler.create_eval_sampler(predicted_entities_dataset, self.args.eval_batch_size, self.args.max_span_size,
                                                    input_reader.context_size, truncate=False, relations_only=True)
        sequences = []
        entities = []
        relations = []
        with torch.no_grad():
            model.eval()
            print("evaluating...")
            print("DEVICE:", self._device)
            # iterate batches
            total = math.ceil(predicted_entities_dataset.document_count / self.args.eval_batch_size)
            batch_count = 0
            for batch in tqdm(sampler, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = batch.to(self._device)
                batch_size = batch.encodings.shape[0]

                # run model (forward pass)
                entity_clf, rel_clf, rels = model(batch.encodings, batch.ctx_masks, batch.entity_masks,
                                                  batch.entity_sizes, input_reader.entity_type_count, batch.entity_spans, 
                                                  batch.entity_sample_masks, evaluate=True)

                # evaluate batch
                # get maximum activation (index of predicted entity type)
                batch_entity_types = entity_clf.argmax(dim=-1)
                # apply entity sample mask
                batch_entity_types *= batch.entity_sample_masks.long()
                batch_entities, batch_relations = evaluator.eval_batch(entity_clf, rel_clf, rels, batch, return_conversions=True)
                for i in range(batch_size):
                    sequences.append(sampler._batches[batch_count][1][i].actual_tokens)
                    entities.append(batch_entities[i])
                    relations.append(batch_relations[i])
                batch_count += 1
        
        if not inference_only:
            global_iteration = epoch * updates_epoch + iteration
            ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
            self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                        epoch, iteration, global_iteration, eval_dataset.label)

            if self.args.store_examples:
                evaluator.store_examples()

        return ner_eval, rel_eval, (sequences, entities, relations)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})