import subprocess

for framework in ["speer", "spert"]:
    for dataset in ["semeval2017_task10", "conll03"]:
        for model in ["map", "mlp", "bilstm", "transformer"]:
            for run in [1, 2, 3]:
                print("Evaluating {} {} on {} (run {})".format(framework, model, dataset, run)) 
                subprocess.run(["python", "main.py", "eval", 
                                "--config", "configs/{}/{}/{}_eval.conf".format(framework, model, dataset),
                                "--model_path", "data/{}/save/{}_eval_{}/run{}/".format(framework, dataset, model, run)])