import json
from tqdm import tqdm
import evaluate
from collections import Counter

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def f1_score(pred, ref):
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


with open("tmp_test/meta-llama-3-8b-instruct_128/narrativeqa/streamingllm.json","r") as fin, \
     open("res/streamingllm.jsonl","w") as fout:

    for line in tqdm(fin):
        data = json.loads(line)

        pred = data["pred"].strip()
        refs = [r.strip() for r in data["answers"]]   # 多个参考答案

        # ========================
        # F1 (取最大)
        # ========================
        f1 = max(f1_score(pred, ref) for ref in refs)

        # ========================
        # ROUGE-L (取最大)
        # ========================
        rouge_scores = []
        for ref in refs:
            rouge_res = rouge.compute(
                predictions=[pred],
                references=[ref]
            )
            rouge_scores.append(rouge_res["rougeL"])

        rouge_l = max(rouge_scores)

        # ========================
        # BERTScore (取最大)
        # ========================
        bert_res = bertscore.compute(
            predictions=[pred] * len(refs),
            references=refs,
            lang="en"
        )

        bert_f1 = max(bert_res["f1"])

        result = {
            "id": data["_id"],
            "F1": float(f1),
            "ROUGE-L": float(rouge_l),
            "BERTScore": float(bert_f1)
        }

        fout.write(json.dumps(result) + "\n")
