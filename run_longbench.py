import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import time
import threading
import subprocess
import re
import evaluate
from collections import Counter

# GPU 监控类
class GPUMonitor:
    def __init__(self, interval=0.01, gpu_id=0):
        """
        interval: 采样间隔（秒），0.01 = 10ms
        gpu_id: GPU 编号
        """
        self.interval = interval
        self.gpu_id = gpu_id

        self.mem_samples = []
        self.power_samples = []

        self._stop_event = threading.Event()
        self._thread = None

    def _get_power(self):
        try:
            base = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"

            with open(f"{base}/in1_input", "r") as f:
                mv = float(f.read().strip())   # mV

            with open(f"{base}/curr1_input", "r") as f:
                ma = float(f.read().strip())   # mA

            power_w = (mv * ma) / 1e6
            return power_w
        except Exception:
            return None


    def _sample_loop(self):
        while not self._stop_event.is_set():
            # 显存（bytes）
            mem = torch.cuda.memory_allocated()
            self.mem_samples.append(mem)

            # 功耗（W）
            power = self._get_power()
            if power is not None:
                self.power_samples.append(power)

            time.sleep(self.interval)

    def start(self):
        self.mem_samples.clear()
        self.power_samples.clear()
        self._stop_event.clear()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        self.start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.start()

    def stop(self):
        torch.cuda.synchronize()
        self.end_time = time.time()

        self._stop_event.set()
        self._thread.join()

    def summary(self, index, token_latencies=None, time_to_first_token=None, kv_cache_size_mb=None):
        duration = self.end_time - self.start_time

        mem_avg = sum(self.mem_samples) / len(self.mem_samples)
        mem_peak = torch.cuda.max_memory_allocated()

        power_avg = (
            sum(self.power_samples) / len(self.power_samples)
            if self.power_samples else None
        )
        power_peak = max(self.power_samples) if self.power_samples else None

        if token_latencies is not None:
            latency = (
                sum(token_latencies[1:]) / (len(token_latencies)-1)
                if len(token_latencies) > 1 else None
            )
        else:
            latency = None
        
        return {
            "sample_index": index,
            "time_sec": duration,

            # 显存
            "mem_avg_MB": mem_avg / 1024 / 1024,
            "mem_peak_MB": mem_peak / 1024 / 1024,

            # 功耗
            "power_avg_W": power_avg,
            "power_peak_W": power_peak,

            # KV缓存大小
            "kv_cache_size_mb": kv_cache_size_mb,

            # 延迟
            "token_latency_sec": token_latencies,
            "time_per_output_token": latency,
            "time_to_first_token": time_to_first_token,

        }


datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

# model2maxlen = {
#     "Llama-2-7b-chat-hf": 3950,
#     "Llama-3-8B-Instruct": 7950,
#     "Meta-Llama-3-70B-Instruct": 7950,
#     "Meta-Llama-3-8B-Instruct-32k": 31500,
#     "Llama-2-7B-32K-Instruct": 31500,
#     "Mistral-7B-Instruct-v0.2": 31500,
#     "Mistral-7B-Instruct-v0.1": 31500,

# }

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

import time
import torch

def get_kv_cache_size_mb(past_key_values):
    total_bytes = 0

    for layer_kv in past_key_values:
        k, v = layer_kv

        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()

    return total_bytes / (1024 ** 2)

@torch.no_grad()
def generate_none_quant(
    i,
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id,
    output_attentions=False,
):

    device = input_ids.device
    batch_size = input_ids.size(0)
    assert batch_size == 1, "This implementation assumes batch_size = 1"

    # =======================
    # Prefill
    # =======================
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=output_attentions,
    )

    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # [1, vocab]

    # =======================
    # Stats containers
    # =======================
    generated_tokens = []
    token_latencies = []
    token_mem = []
    token_mem_reserved = []

    # 当前 attention_mask（decode 时必须增长）
    cur_attention_mask = attention_mask

    torch.cuda.reset_peak_memory_stats()

    # =======================
    # Decode loop
    # =======================
    for step in range(max_new_tokens):
        start = time.perf_counter()

        # greedy
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

        # 1️⃣ attention_mask 增长
        cur_attention_mask = torch.cat(
            [
                cur_attention_mask,
                torch.ones(
                    (1, 1),
                    device=device,
                    dtype=cur_attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        # 2️⃣ position_ids 必须显式给
        # 等价于 HF generate 的做法
        position_ids = torch.tensor(
            [[cur_attention_mask.shape[1] - 1]],
            device=device,
            dtype=torch.long,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=cur_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
        )

        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        token_latencies.append(latency)
        if step == 0:
            first_token_latency = latency
        token_mem.append(torch.cuda.memory_allocated())
        token_mem_reserved.append(torch.cuda.memory_reserved())

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        generated_tokens.append(next_token)

        kv_size = get_kv_cache_size_mb(past_key_values)
        print(f"Step {step} KV size: {kv_size:.2f} MB")

        # HF generate 的停止条件
        if next_token.item() in eos_token_id:
            break

    # =======================
    # 拼接输出（与 HF generate 一致）
    # =======================
    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, dim=1)
        sequences = torch.cat([input_ids, generated_tokens], dim=1)
    else:
        sequences = input_ids

    kv_cache_size_mb = get_kv_cache_size_mb(past_key_values)

    return {
        "sequences": sequences,
        "time_to_first_token": first_token_latency + prefill_time,
        "token_latencies": token_latencies,
        "token_mem": token_mem,
        "token_mem_reserved": token_mem_reserved,
        "mem_peak": torch.cuda.max_memory_allocated(),
        "mem_reserved_peak": torch.cuda.max_memory_reserved(),
        "kv_cache_size_mb": kv_cache_size_mb,
    }


from transformers.cache_utils import (
    QuantizedCache,
    QuantizedCacheConfig,
)


def tensor_nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    return 0

def recursive_nbytes(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # torch tensor
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()

    # basic containers
    if isinstance(obj, dict):
        return sum(recursive_nbytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(recursive_nbytes(v, seen) for v in obj)

    # objects with __dict__
    if hasattr(obj, "__dict__"):
        return sum(recursive_nbytes(v, seen) for v in vars(obj).values())

    return 0


def get_cache_storage_bytes(past_key_values):
    out = {}

    # 1) 统计整个对象递归包含的 tensor bytes
    out["total_recursive_bytes"] = recursive_nbytes(past_key_values)

    # 2) 单独看 residual cache
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None or value_cache is not None:
        out["residual_bytes"] = recursive_nbytes(key_cache) + recursive_nbytes(value_cache)

    # 3) 单独看 quantized cache
    qk = getattr(past_key_values, "_quantized_key_cache", None)
    qv = getattr(past_key_values, "_quantized_value_cache", None)
    if qk is not None or qv is not None:
        out["quantized_bytes"] = recursive_nbytes(qk) + recursive_nbytes(qv)

    return out

@torch.no_grad()
def generate_quant_bench(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id,
    cache_config,
    output_attentions=False,
):
    """
    Quantized KV cache benchmark (single generate call)

    Measures:
        - TTFT (time to first token)
        - TPOT (per token latency)
        - GPU memory usage

    Assumes:
        batch_size == 1
    """

    device = input_ids.device
    assert input_ids.size(0) == 1, "This implementation assumes batch_size = 1"

    # =======================
    # Streamer
    # =======================
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    token_latencies = []
    token_mem = []
    token_mem_reserved = []

    # =======================
    # Reset GPU memory stats
    # =======================
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        min_length=input_ids.shape[1] + 1,
        eos_token_id=eos_token_id,
        output_attentions=output_attentions,
        streamer=streamer,
        cache_implementation="quantized",
        cache_config=cache_config,
        return_dict_in_generate=True,
    )

    result_container = {}

    def run_generate():
        result_container["output"] = model.generate(**generation_kwargs)

    # =======================
    # Start generation
    # =======================
    generate_start = time.perf_counter()
    thread = threading.Thread(target=run_generate)
    thread.start()

    first_token_latency = None
    last_time = None

    # =======================
    # Stream tokens
    # =======================
    for _ in streamer:
        torch.cuda.synchronize()
        now = time.perf_counter()

        if first_token_latency is None:
            # TTFT
            first_token_latency = now - generate_start
            # token_latencies.append(first_token_latency)
        else:
            # TPOT
            latency = now - last_time
            token_latencies.append(latency)

        token_mem.append(torch.cuda.memory_allocated())
        token_mem_reserved.append(torch.cuda.memory_reserved())

        last_time = now

    thread.join()

    out = result_container["output"]
    pkv = out.past_key_values

    stats = get_cache_storage_bytes(pkv)
    # print(type(pkv))
    values = []
    for k, v in stats.items():
        values.append(v / 1024**2)
        print(f"{k}: {v / 1024**2:.2f} MB")
   
    sequences = result_container["output"].sequences

    # print(f"token_memory:{[mem / 1024 / 1024 for mem in token_mem]} MB")
    # print(type(result_container["output"].past_key_values))
    # print(result_container["output"].past_key_values)
    return {
        "sequences": sequences,
        "time_to_first_token": first_token_latency,
        "token_latencies": token_latencies,
        "token_mem": token_mem,
        "token_mem_reserved": token_mem_reserved,
        "mem_peak": torch.cuda.max_memory_allocated(),
        "mem_reserved_peak": torch.cuda.max_memory_reserved(),
        "kv_cache_size_mb": stats["total_recursive_bytes"] / 1024**2,
    }


def reset_pyramidkv_state(model):
    for layer in model.model.layers:
        attn = layer.self_attn

        # 只 reset 你自己加的状态
        if hasattr(attn, "kv_seq_len"):
            attn.kv_seq_len = 0

        if hasattr(attn, "step"):
            attn.step = 0

        if hasattr(attn, "prefill_phase"):
            attn.prefill_phase = True

        # 如果 kv_cluster 有状态，也要清
        if hasattr(attn, "kv_cluster"):
            if hasattr(attn.kv_cluster, "kv_seq_len"):
                attn.kv_cluster.kv_seq_len = 0

            if hasattr(attn.kv_cluster, "seen_tokens"):
                attn.kv_cluster.seen_tokens = 0

def reset_kv_strategy_state(model):
    for layer in model.model.layers:
        attn = layer.self_attn

        if hasattr(attn, "kv_seq_len"):
            attn.kv_seq_len = 0

        for attr in [
            "selected_idx",
            "score_cache",
            "hh_score",
            "kv_cluster",
            "past_key_value",
            "key_cache",
            "value_cache",
        ]:
            if hasattr(attn, attr):
                setattr(attn, attr, None)

        if hasattr(attn, "prefill_done"):
            attn.prefill_done = False

        if hasattr(attn, "decode_step"):
            attn.decode_step = 0

def main(args):
    

    print("Loading data...")
    # rouge = evaluate.load("rouge")
    # bertscore = evaluate.load("bertscore")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            

    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
                
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)

    # fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
    save_directory = f"./res/{args.dataset}/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if args.quant_method is None:
        file = open(os.path.join(save_directory, f"{args.method}-{args.max_capacity_prompts}.jsonl"), "w")
        fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, \
                                 f"{args.method}-{args.max_capacity_prompts}.jsonl"), "w")
    else:
        if args.method is not None:
            file = open(os.path.join(save_directory, f"{args.method}-{args.quant_method}-{args.nbits}b-{args.max_capacity_prompts}.jsonl"), "w")
            fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, \
                                     f"{args.method}-{args.quant_method}-{args.nbits}b-{args.max_capacity_prompts}.jsonl"), "w")
        else:
            file = open(os.path.join(save_directory, f"{args.quant_method}-{args.nbits}b-{args.max_capacity_prompts}.jsonl"), "w")
            fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, \
                                     f"{args.quant_method}-{args.nbits}b-{args.max_capacity_prompts}.jsonl"), "w")
            
     
    try:
        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            if i >= 50 :
                break

            batch_prompts = prompts[i:i+args.eval_batch_size]
            batch_inputs = inputs[i:i+args.eval_batch_size]
            batch_contexts = contexts[i:i+args.eval_batch_size]
            batch_answerss = answerss[i:i+args.eval_batch_size]
            batch_lengths = lengths[i:i+args.eval_batch_size]
            
            batch_datasets = datasets[i:i+args.eval_batch_size]
            batch_languages = languages[i:i+args.eval_batch_size]
            batch_all_classess = all_classess[i:i+args.eval_batch_size]
            batch__ids = _ids[i:i+args.eval_batch_size]
            
            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

            if len(batch_input_ids[0]) > model_max_len:
                half = int(model_max_len/2)
                prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
                
                tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
                batch_input_ids = tokenized_prompts.input_ids
                attention_mask = tokenized_prompts.attention_mask

            # # default to True
            # if args.method == "DynamicKV":
            #     args.output_attentions = True
            # else:
            #     args.output_attentions=False

            if args.max_capacity_prompts != -1:
                max_capacity_prompts = args.max_capacity_prompts
            elif args.max_capacity_prompts_ratio != -1:
                max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
            
            
            if args.method is not None and args.method.lower() != "fullkv":
                if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "adakv", "headkv", "think"]:
                    window_sizes = 8
                elif args.method.lower() in ["streamingllm"]:
                    window_sizes = max_capacity_prompts - 4

                if args.method.lower() =='headkv':
                    with open(args.head_path, 'r') as file:
                        head_list = json.loads(file.readline())
                    head_score_list = [np.mean(l[1]) for l in head_list.items()]
                    head_score_list = torch.tensor(head_score_list / sum(head_score_list))
                    total_attention = head_score_list.reshape(model.config.num_hidden_layers, model.config.num_attention_heads)
                    total_pool_capacity = (args.max_capacity_prompts // args.head_beta) * model.config.num_hidden_layers * model.config.num_attention_heads
                    min_num = (args.max_capacity_prompts - args.max_capacity_prompts // args.head_beta)
                    head_capacity = torch.round(total_attention * total_pool_capacity + min_num).int()
                    model.model.config.head_capacity = head_capacity    

                kernel_sizes = 7
                pooling = "maxpool"
                ratio = args.pruning_ratio
                recent_size = args.recent_size

                layers = len(model.model.layers)
                # check if window_sizes is a list
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                if not isinstance(ratio, list):
                    ratio = [ratio] * layers
                if not isinstance(recent_size, list):
                    recent_size = [recent_size] * layers
                for layer_idx in range(layers):
                    model.model.layers[layer_idx].self_attn.config.window_size = window_sizes[layer_idx]
                    model.model.layers[layer_idx].self_attn.config.max_capacity_prompt = max_capacity_prompts[layer_idx]
                    model.model.layers[layer_idx].self_attn.config.kernel_size = kernel_sizes[layer_idx]
                    model.model.layers[layer_idx].self_attn.config.pooling = pooling
                    model.model.layers[layer_idx].self_attn.config.merge = args.merge
                    model.model.layers[layer_idx].self_attn.config.floor = args.floor
                    model.model.layers[layer_idx].self_attn.config.ratio = ratio[layer_idx]
                    model.model.layers[layer_idx].self_attn.config.recent_size = recent_size[layer_idx]

            reset_kv_strategy_state(model)
            
            # 创建并启动 GPU 监控
            monitor = GPUMonitor(interval=0.01, gpu_id=0)
            monitor.start()

            context_length = batch_input_ids.shape[-1]
            if args.quant_method == None:        
                # output = model.generate(
                #     **tokenized_prompts,
                #     output_attentions = args.output_attentions,
                #     max_new_tokens=output_max_len,
                #     num_beams=1,
                #     do_sample=False,
                #     temperature=1.0,
                #     min_length=context_length+1,
                #     eos_token_id=[tokenizer.eos_token_id]
                # )
                gen_out = generate_none_quant(
                    i,
                    model=model,
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=output_max_len,
                    eos_token_id=[tokenizer.eos_token_id],
                    output_attentions=args.output_attentions,
                )
                
                output = gen_out["sequences"]
                token_latencies = gen_out["token_latencies"]
                time_to_first_token = gen_out["time_to_first_token"]
                kv_cache_size_mb = gen_out["kv_cache_size_mb"]
            else:
                # output = model.generate(
                #     **tokenized_prompts,
                #     output_attentions = args.output_attentions,
                #     max_new_tokens=output_max_len,
                #     num_beams=1,
                #     do_sample=False,
                #     temperature=1.0,
                #     min_length=context_length+1,
                #     eos_token_id=[tokenizer.eos_token_id],
                #     cache_implementation="quantized", 
                #     cache_config={"nbits": args.nbits, "backend": "HQQ","device":"cuda","residual_length":output_max_len,"axis_key":1,"q_group_size":64},
                # )
                gen_out = generate_quant_bench(
                    model=model,
                    tokenizer=tokenizer,  # ⚠️ 多了这个参数
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=output_max_len,
                    eos_token_id=[tokenizer.eos_token_id],
                    cache_config={
                        "nbits": args.nbits,
                        "backend": "HQQ",
                        "device": "cuda",
                        "residual_length": output_max_len,
                        "axis_key": 1,
                        "q_group_size": 64,
                    },
                    output_attentions=args.output_attentions,
                )

                output = gen_out["sequences"]
                token_latencies = gen_out["token_latencies"]
                time_to_first_token = gen_out["time_to_first_token"]
                kv_cache_size_mb = gen_out["kv_cache_size_mb"]


            # 停止 GPU 监控
            monitor.stop()
            # pred = batch_outputs[0].strip()
            # refs = batch_answerss[0][0]
            # print(f"pred: {pred}")
            # print(f"refs: {refs}")

            # f1 = max_f1(pred, refs)
            # rouge_result = rouge.compute(
            #     predictions=[pred],
            #     references=[refs[0]]
            # )
            # rouge_l = rouge_result["rougeL"]

            # bert_result = bertscore.compute(
            #     predictions=[pred],
            #     references=[refs[0]],
            #     lang="en"
            # )
            # bert_f1 = bert_result["f1"][0]

            sample_idx = i // args.eval_batch_size
            stats = monitor.summary(sample_idx, token_latencies, time_to_first_token, kv_cache_size_mb)
            # stats = monitor.summary(sample_idx)
            file.write(f"{json.dumps(stats)}\n")

            batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
            
            # print(f"debbug batch_outputs {batch_outputs}")
            
            batch_generations = batch_outputs

            # torch.cuda.empty_cache()

            for j in range(args.eval_batch_size):
                
                example = {}
                
                example["prompt"] = batch_prompts[j]
                example["input"] = batch_inputs[j]
                example["context"] = batch_contexts[j]
                example["answers"] = batch_answerss[j]
                example["pred"] = batch_generations[j]
                example["length"] = batch_lengths[j]
                
                example["dataset"] = batch_datasets[j]
                example["language"] = batch_languages[j]
                example["all_classes"] = batch_all_classess[j]
                example["_id"] = batch__ids[j]

                # print(f'{batch_generations[j]}')
                fout.write(json.dumps(example) + "\n")
            
            torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
    finally:
        fout.close()
        file.close()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument('--head_path', type=str, default='./data/heads_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json', help='Path to head score (HeadKV)')
    parser.add_argument('--head_beta', type=float, default=1.01, help='hyper-parameter used on HeadKV')
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")

    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    if args.quant_method == "kvquant":
        from pyramidkv.quantcache import KVQuantizedCache
        from transformers import cache_utils
        cache_utils.HQQQuantizedCache = KVQuantizedCache
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from pyramidkv.monkeypatch import replace_llama,replace_mistral
    # if args.method and args.method.lower() != "fullkv":
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()
    
    save_dir = args.save_dir
    
        
    max_capacity_prompts = args.max_capacity_prompts
    
    for idx, dataset in enumerate(datasets):
        if dataset not in ["narrativeqa", "hotpotqa", "multi_news", "passage_retrieval_en", "musique", "lcc"]:
            continue
        # if dataset != "narrativeqa": 
            # continue
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        
        args.dataset = dataset
        
        args.data_file = f"data/LongBench/{args.dataset}.jsonl"
        
        main(args)
        torch.cuda.empty_cache()

