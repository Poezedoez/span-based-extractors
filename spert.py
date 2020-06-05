import argparse

from args import train_argparser, eval_argparser, infer_argparser, map_args
from config_reader import process_configs, process_configs_serial
from model import input_reader
from model.trainers import SpERTTrainer, SpEERTrainer
from model import util

def __train(run_args, queue=None):
    trainer = get_trainer(run_args.model_type)(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args, queue=None):
    trainer = get_trainer(run_args.model_type)(run_args)
    trainer.eval(train_path=run_args.train_path, eval_path=run_args.eval_path, 
                 types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)

def load_inference_model(args):
    arg_parser = infer_argparser()
    mapped_args = map_args(arg_parser, args)
    run_args = process_configs_serial(arg_parser, args=mapped_args)
    model = get_trainer(run_args.model_type)(run_args)
    print("SpERT ready for inference.")
    
    return model

def _infer(document):
    """
    When called from command line.
    This function is for testing only, because it 
    loads the model at every inference call
    """
    arg_parser = infer_argparser()
    run_args = process_configs_serial(arg_parser)
    model = get_trainer(run_args.model_type)(run_args)
    print("SpERT ready for inference.")
    
    infer(model, document, run_args.types_path)

def infer(model, document, types, verbose=False):
    """
    Do inference with the model on the given document.
    """
    raw_output = model.infer(document_data=document, types_path=types,
                 input_reader_cls=input_reader.StringInputReader)
    json_format = util.convert_to_json_dataset(raw_output)
    if verbose:
        _print_inference_results(json_format)

def _print_inference_results(json_data):
    print("Converted json output:")
    for sentence in json_data:
        print(sentence["tokens"])
        print(sentence["entities"])
        print(sentence["relations"])
        print()

_TRAINERS = {
    # 'spert': SpERTTrainer,
    # 'spet': SpERTTrainer,
    'speer':SpEERTrainer
}

def get_trainer(name, default=SpERTTrainer):
    return _TRAINERS.get(name, default)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()
    example_data = {
        'guid': 'IDtest',
        'sentences': [
            'In contrast with the normal auto-encoder, denoising auto-encoder (Vincent etal., 2010) could improve the model learning ability by introducing noise in the form of random tokendeleting and swapping in this input sentence',
            'Neural machine translations (NMT) (Bahdanau et al., 2015; Vaswani et al., 2017) have set several state-of-the-art  new  benchmarks  (Bojar  et  al.,  2018;  Barrault  et  al.,  2019)',
            'Our empirical results showthat the UNMT model outperformed the SNMT model, although both of their performances decreasedsignificantly  in  this  scenario.'
        ]
    }
    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'infer':
        _infer(example_data)
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python spert.py train ...'")
