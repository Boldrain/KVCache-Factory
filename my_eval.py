import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = [
        "narrativeqa",
        "hotpotqa",
        "musique",
        "multi_news",
        "passage_retrieval_en",
        "lcc",
        ]
    
    quants_list = [
        "none",
        "kivi",
        "kvquant",
        ]
    
    results_list = [[
        ["dataset"],
        ["FullKV"],
        ["SnapKV"],
        ["StreamingLLM"],
        ["H2O"],
        ["PyramidKV"],
    ] for _ in range(len(quants_list))]
    
# /home/zry/zry/KVCache-Factory/tmp_test/meta-llama-3-8b-instruct_128
    for quant_idx, quant in enumerate(quants_list):
        for dataset in dataset_list:
            
            results_list[quant_idx][0].append(dataset)
            
            for idx, method in enumerate(["FullKV", "SnapKV", "StreamingLLM", "H2O", "PyramidKV"]):
            # for idx, method in enumerate(["H2_global", "PyramidKV_global", "local"]):
                try:
                    args.method = method
                    args.dataset = dataset
                    if quant == "none":
                        args.eval_file = os.path.join(args.results_dir,dataset,f"{method}-128.jsonl".lower())
                    else :
                        args.eval_file = os.path.join(args.results_dir,dataset,f"{method}-{quant}-8b-128.jsonl".lower())
                    
                    
                    # try:
                    
                    scores = dict()
                    # if args.longbench_e:
                    #     path = f"pred_e/{args.model}/"
                    # else:
                    #     path = f"pred_e/{args.model}/"
                    # all_files = os.listdir(path)
                    # print("Evaluating on:", all_files)
                    
                    # for filename in all_files:
                        # if not filename.endswith("jsonl"):
                        #     continue
                    predictions, answers, lengths = [], [], []
                    # dataset = filename.split('.')[0]
                    with open(args.eval_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                predictions.append(data["pred"])
                                answers.append(data["answers"])
                                all_classes = data["all_classes"]
                                if "length" in data:
                                    lengths.append(data["length"])
                            except:
                                print("error")
                    if args.longbench_e:
                        score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    else:
                        score = scorer(args.dataset, predictions, answers, all_classes)
                        if args.dataset == 'qasper':
                            score_e = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    scores[args.dataset] = score
                        # if dataset == 'qasper':
                        #     scores[dataset + '_e'] = score_e
                        
                    # if args.longbench_e:
                    #     out_path = f"H2O/results/{args.model}/result.json"
                    # else:
                    #     out_path = f"H2O/results/{args.model}/result.json"
                        # out_path_e = f"pred/{args.model}/result_e.json"
                        # with open(out_path_e, "w") as f:
                        #     json.dump(score_e, f, ensure_ascii=False, indent=4)
                        
                    # output_dir = "/home/zry/zry/KVCache-Factory/res/"
                    
                    results_list[quant_idx][idx+1].append(score)
                    
                    # with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    #     json.dump(scores, f, ensure_ascii=False, indent=4)
                
                    print(f"dataset {args.dataset} method {args.method} scores {scores}")
                except:
                    
                    results_list[quant_idx][idx+1].append(-1)
                    
                    print(f"dataset {args.dataset} method {args.method} scores {None}")
                    
    import os
    import csv

    # quant_names = ["none", "kivi", "kvquant"]
    output_dir = "/home/zry/zry/KVCache-Factory/res"

    for quant_idx, quant_name in enumerate(quants_list):
        output_file = os.path.join(output_dir, f"results_{quant_name}.csv")
        with open(output_file, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerows(results_list[quant_idx])
