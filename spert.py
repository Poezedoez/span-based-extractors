import argparse

from args import train_argparser, eval_argparser
from config_reader import process_configs
from spert import input_reader
from spert.trainers import SpERTTrainer


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)

def __infer(run_args, example_data):
    trainer = SpERTTrainer(run_args)
    # Document data is a dictionary with the guid and the sentences of a document
    entity_clfs = trainer.infer(document_data=example_data, types_path=run_args.types_path,
                 input_reader_cls=input_reader.StringInputReader)

    print(entity_clfs)


def _infer(example_data):
    arg_parser = eval_argparser()
    process_configs(target=__infer, arg_parser=arg_parser, data=example_data)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()
    example_data = {
        'guid': 'test',
        'sentences': [
            'In contrast with the normal auto-encoder, denoising auto-encoder (Vincent etal., 2010) could improve the model learning ability by introducing noise in the form of random tokendeleting and swapping in this input sentence',
            'Neural machine translations (NMT) (Bahdanau et al., 2015; Vaswani et al., 2017) have set several state-of-the-art  new  benchmarks  (Bojar  et  al.,  2018;  Barrault  et  al.,  2019)',
            'Our empirical results showthat the UNMT model outperformed the SNMT model, although both of their performances decreasedsignificantly  in  this  scenario.'
        ] * 100
    }
    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'infer':
        _infer(example_data)
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python spert.py train ...'")
