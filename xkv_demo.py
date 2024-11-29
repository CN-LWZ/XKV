import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from datetime import datetime
import json
import random
import argparse
from xkv.generate_xkv import generate,sample
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import transformers
from xkv.llama_model_xkv import LlamaForCausalLM
from transformers import AutoTokenizer
from xkv.xkv_utils import XKV_Var
from xkv.xkv_utils import XKV_miniprefill,XKV_generate



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


def main(args):
    

    print("Loading data...")
    
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

    os.makedirs(os.path.join(args.save_dir, f"{model_name}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}", args.dataset, f"{args.method}.json"), "w")

    count=0
    sample_count=len(prompts)*args.sampling
    sample_list=[]
    allocation_list=[]
    
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
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

        if len(batch_input_ids[0]) > model_max_len:#Truncation
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask


        layers = len(model.model.layers)

        context_length = batch_input_ids.shape[-1]    
        
        if count<=sample_count:    
            if count<sample_count:
                allocation_list=XKV_miniprefill(model,
                                                tokenizer,
                                                tokenized_prompts,
                                                output_max_len,
                                                context_length,
                                                args,layers)
                
                sample_list.append(allocation_list)
                count+=1
            else:
                allocation_list=np.array(sample_list)
                allocation_list=np.mean(allocation_list,axis=0)
                allocation_list=np.round(allocation_list).astype(int)
                count+=1

        output=XKV_generate(model,tokenizer,
                            tokenized_prompts,
                            output_max_len,
                            context_length,args,
                            layers,
                            allocation_list)

        batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
        
        batch_generations = batch_outputs

        torch.cuda.empty_cache()
        
        
   
        for j in range(args.eval_batch_size):#写进文件
            
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


            fout.write(json.dumps(example) + "\n")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    
    parser.add_argument("--total_size", type=int, default=0, help="")
    parser.add_argument("--set_value", type=float, default=0.7, help="")
    parser.add_argument("--sampling", type=float, default=0.1, help="")
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--model_path", type=str, default="/home/users/lwz/mymodel/Meta-Llama-3-8B-Instruct/")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", help="Currently, only flash_attention_2 is supported.")
    parser.add_argument("--method", type=str,  default="XKV")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    datasets = ["narrativeqa"]

    dataset2maxlen = {
        "narrativeqa": 128,
    }

    model2prompt = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    }

    model2maxlen = {
        "llama3": 7950,
        "llama-3": 7950,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


        
    transformers.generation.utils.GenerationMixin.generate=generate
    transformers.generation.utils.GenerationMixin._sample=sample
    model = LlamaForCausalLM.from_pretrained(  
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
 
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on dataset {dataset}")
        
        args.dataset = dataset
        
        args.data_file = f"data/{args.dataset}.jsonl"
        
        main(args)
