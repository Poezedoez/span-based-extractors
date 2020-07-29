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
        # for dataset in [args.dataset] if args.dataset else ["semeval2017_task10", "conll03", "za"]:
        for dataset in [args.dataset] if args.dataset else ["semeval2017_task10", "conll03"]:
            for model in [args.model] if args.model else ["map", "mlp", "bilstm", "transformer"]:
                for run in [1,2,3]:
                    print("Training {} {} on {} (run {})".format(framework, model, dataset, run)) 
                    subprocess.run(["python", "main.py", "train", 
                                    "--config", "configs/{}/{}_train.conf".format(framework, dataset),
                                    "--save_path", "data/{}/save/{}_train_{}/run{}/".format(framework, dataset, model, run),
                                    "--log_path", "data/{}/log/{}_train_{}/run{}/".format(framework, dataset, model, run),                                  
                                    "--model_type", framework,
                                    "--feature_enhancer", model,
                                    "--label", "{}_train_{}".format(dataset, model)])    


def run_eval(args):
    for framework in [args.framework] if args.framework else ["spert", "speer"]:
        for dataset in [args.dataset] if args.dataset else ["semeval2017_task10", "conll03"]:
            for model in [args.model] if args.model else ["map", "mlp", "bilstm", "transformer"]:
                for run in [1, 2, 3]:
                    print("Evaluating {} {} on {} (run {})".format(framework, model, dataset, run)) 
                    subprocess.run(["python", "main.py", "eval", 
                                    "--config", "configs/{}/{}_eval.conf".format(framework, dataset),
                                    "--log_path", "data/{}/log/{}_train_{}/run{}/".format(framework, dataset, model, run), 
                                    "--model_path", "data/{}/save/{}_train_{}/run{}/".format(framework, dataset, model, run),
                                    "--model_type", framework,
                                    "--feature_enhancer", model,
                                    "--label", "{}_eval_{}".format(dataset, model)])


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
    arg_parser.add_argument('--framework', type=str, help="Framework: spert, speer, sprt. Defaults to all", default=None)
    arg_parser.add_argument('--run', type=str, help="Run: 1, 2, 3. (eval and read mode) Defaults to all", default=None)

    args = arg_parser.parse_args()
    if args.mode == "eval":
        run_eval(args)
    if args.mode == "train":
        run_train(args)
    if args.mode == "read":
        run_read(args)
    else:
        print("Unknown mode: {}".format(args.mode))