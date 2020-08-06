import subprocess
import argparse
import glob
import pandas as pd
import numpy as np
import os
import statistics
from collections import defaultdict


def run_train(args):
    for framework in [args.framework] if args.framework else ["spert", "speer", "sprt"]:
        for dataset in [args.dataset] if args.dataset else ["za1", "za2", "za3"]:
            for ratio in [args.ratio] if args.ratio else [0.25, 0.5, 0.75]:
                for model in [args.model] if args.model else ["bilstm", "transformer"]:
                    for run in [args.run] if args.run else [1, 2, 3]:
                        print("Training {} {} on {} (run {})".format(framework, model, dataset, run)) 
                        subprocess.run(["python", "main.py", "train", 
                                        "--config", "configs/{}/za_train.conf".format(framework),
                                        "--train_path", "data/datasets/{}/za_train_{}.json".format(dataset, ratio),
                                        "--valid_path", "data/datasets/{}/za_dev_{}.json".format(dataset, ratio),
                                        "--types_path", "data/datasets/{}/za_types.json".format(dataset),
                                        "--save_path", "data/{}/save/{}_train_{}_{}/run{}/".format(framework, dataset, model, ratio, run),
                                        "--log_path", "data/{}/log/{}_train_{}_{}/run{}/".format(framework, dataset, model, ratio, run),                                  
                                        "--model_type", framework,
                                        "--feature_enhancer", model,
                                        "--label", "{}_train_{}_{}".format(dataset, model, ratio)])    


def run_eval(args):
    for framework in [args.framework] if args.framework else ["spert", "speer", "sprt"]:
        for dataset in [args.dataset] if args.dataset else ["za1", "za2", "za3"]:
            for ratio in [args.ratio] if args.ratio else [0.25, 0.5, 0.75]:
                for model in [args.model] if args.model else ["bilstm", "transformer"]:
                    for run in [args.run] if args.run else [1, 2, 3]:
                        print("Evaluating {} {} on {} (run {})".format(framework, model, dataset, run)) 
                        subprocess.run(["python", "main.py", "eval", 
                                        "--config", "configs/{}/za_eval.conf".format(framework),
                                        "--log_path", "data/{}/log/{}_eval_{}_{}/run{}/".format(framework, dataset, model, ratio, run), 
                                        "--predicted_entities_path", "data/sprt/save/predictions_{}_{}.json".format(model, ratio),
                                        "--model_path", "data/{}/save/{}_train_{}_{}/run{}/".format(framework, dataset, model, ratio, run),
                                        "--types_path", "data/datasets/{}/za_types.json".format(dataset),
                                        "--model_type", framework,
                                        "--feature_enhancer", model,
                                        "--label", "{}_eval_{}_{}".format(dataset, model, ratio)])


def run_read(args):
    evaluations = defaultdict(lambda: defaultdict())
    for framework in [args.framework] if args.framework else ['spert', 'speer', 'sprt']:
        for dataset in [args.dataset] if args.dataset else ['conll03', 'semeval2017_task10']:
            for architecture in [args.model] if args.model else ['map', 'mlp', 'bilstm', 'transformer']:
                ner_ps, ner_rs, ner_f1s = [], [], []
                rel_ps, rel_rs, rel_f1s = [], [], []
                scores = [ner_ps, ner_rs, ner_f1s, rel_ps, rel_rs, rel_f1s]
                # accumulate over different runs
                for run in [args.run] if args.run else [1,2,3]:
                    score = read_score(framework, dataset, architecture, run)
                    if score:
                        for i, s in enumerate(scores):
                            s.append(score[i]) 
                # calculate margin of error and mean over all different runs
                print("FINAL SCORES {} {} {}:".format(framework, dataset, architecture))
                print()
                s_names = ['ner_precision', 'ner_recall', 'ner_f1', 'rel_precision', 'rel_recall', 'rel_f1']
                for s, s_name in zip(scores, s_names):
                    a = np.array(s)
                    mean = np.mean(a)
                    dif_square = (a-mean)**2
                    std = np.sqrt(np.sum(dif_square)/len(s))
                    error_margin = std/np.sqrt(len(s))
                    print(s_name, mean, error_margin)
                print()


def read_score(framework, dataset, model, run):
    print("-"*100)
    print("Scores for settings: |{}| |{}| |{}| |run {}|".format(framework, dataset, model, run))
    scores_file = "eval_test.csv"
    path = os.path.join("data/{}/log".format(framework), "{}_eval_{}".format(dataset, model), "run{}".format(run), scores_file)

    if not os.path.exists(path):
        print("Evaluation file not found!")
        print()
        return None

    df = pd.read_csv(path, delimiter=";")

    print()
    print(df[['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro', 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro']].to_string(index=False))
    print()

    if not df.empty:
        columns = ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro', 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro']
        values = df[columns].values.tolist()[0]
        print()
        return values
    else:
        print("No evaluation scores found!")
        print()
        return None


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval' or 'read")
    arg_parser.add_argument('--model', type=str, help="Model: map, mlp, bilstm, transformer. Defaults to all", default=None)
    arg_parser.add_argument('--dataset', type=str, help="Dataset: conll03, conll04, semeval2017_task10. Defaults to all", default=None)
    arg_parser.add_argument('--ratio', type=float, help="Ratio of knowledge graph concepts used in train set", default=None)
    arg_parser.add_argument('--framework', type=str, help="Framework: spert, speer, sprt. Defaults to all", default=None)
    arg_parser.add_argument('--run', type=int, help="Run: 1, 2, 3. (eval and read mode) Defaults to all", default=None)

    args = arg_parser.parse_args()
    if args.mode == "eval":
        run_eval(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "read":
        run_read(args)
    else:
        print("Unknown mode: {}".format(args.mode))