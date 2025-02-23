import torch
import numpy as np
import pandas as pd
import json
import fire
import os

#from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# 4.48.0
from transformers import MllamaForConditionalGeneration,MllamaProcessor
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor


from openai import OpenAI

# TODO: 구현하기
from dataset.kmmbench import kmmbench_eval
from dataset.kmmstar import kmmstar_eval
from dataset.kdtcbench import kdtcbench_eval
from dataset.kllavaw import kllavaw_eval

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# NCSOFT/K-MMBench
# NCSOFT/K-MMStar
# NCSOFT/K-DTCBench
# NCSOFT/K-LLaVA-W

# Bllossom/llama-3.2-Korean-Bllossom-AICA-5B
# NCSOFT/VARCO-VISION-14B-HF
# AIDC-AI/Ovis1.6-Gemma2-9B
# gpt-4o-2024-11-20
# AIDC-AI/Ovis2-34B

def main(
        dataset = '...dataset.',
        base_model = '...model...',
        cutoff_len = 2048,
        api_key = '...your_api...'
    ):
    
    login(token='...your_token...')
    
    # Load dataset
    #eval_dataset = load_dataset(dataset)
    
    ## Model loading
    device_map = 'auto'
    
    if ('Ovis' in base_model) or ('Gukbap' in base_model):
        print(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            cache_dir="/data/cache/", # database 폴더로 전송
            torch_dtype=torch.float16,
            trust_remote_code=True,
            multimodal_max_length=cutoff_len # 2048
        )
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
    
    # TODO: evaluation script 만들기
    elif 'gpt' in base_model:
        print(base_model)
        model = base_model # name
    
    elif 'Bllossom' in base_model:
        print(base_model)
        model = MllamaForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            cache_dir='/data/cache',
            device_map=device_map
            )
        processor = MllamaProcessor.from_pretrained(base_model)
        
    elif 'VARCO' in base_model:
        print(base_model)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype="float16",
            device_map=device_map,
            cache_dir='/data/cache',
            #attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(base_model, device_map=device_map)
        
    else:
        raise Exception("Not implementation!!")
    
    ### Evaluation
    # 4329개
    if 'K-MMBench' in dataset:
        eval_dataset = pd.read_parquet("./data/kmmbench/dev-00000-of-00001.parquet")
        if ('Ovis' in base_model) or ('Gukbap' in base_model):
            average = kmmbench_eval(eval_dataset, model, text_tokenizer, visual_tokenizer, 'ovis')
        
        elif 'Bllossom' in base_model:
            average = kmmbench_eval(eval_dataset, model, processor, None, 'bllossom')
        
        elif 'VARCO' in base_model:
            average = kmmbench_eval(eval_dataset, model, processor, None, 'VARCO')
            
        elif 'GPT' in base_model:
            pass
        
        print("### KoMMBench score:", average*100)
    
    # TODO: MMStar 구현하기
    # 1,500개
    elif 'K-MMStar' in dataset:
        eval_dataset = pd.read_parquet("./data/kmmstar/val-00000-of-00001.parquet")
        if ('Ovis' in base_model) or ('Gukbap' in base_model):
            average = kmmstar_eval(eval_dataset, model, text_tokenizer, visual_tokenizer, 'ovis')
        
        elif 'Bllossom' in base_model:
            average = kmmstar_eval(eval_dataset, model, processor, None, 'bllossom')
        
        elif 'VARCO' in base_model:
            average = kmmstar_eval(eval_dataset, model, processor, None, 'VARCO')
            
        elif 'GPT' in base_model:
            pass
        
        print("### K-MMStar score:", average*100)
    
    elif 'K-DTCBench' in dataset:
        eval_dataset = pd.read_parquet("./data/kdtcbench/test-00000-of-00001.parquet")
        
        if ('Ovis' in base_model) or ('Gukbap' in base_model):
            average = kdtcbench_eval(eval_dataset, model, text_tokenizer, visual_tokenizer, 'ovis')
        
        elif 'Bllossom' in base_model:
            average = kdtcbench_eval(eval_dataset, model, processor, None, 'bllossom')
        
        elif 'VARCO' in base_model:
            average = kdtcbench_eval(eval_dataset, model, processor, None, 'VARCO')
            
        elif 'GPT' in base_model:
            pass
        
        print("### K-DTCBench score:", average*100)
    
    elif 'K-LLaVA-W' in dataset:
        eval_dataset = pd.read_parquet("./data/kllavaw/test-00000-of-00001.parquet")
        
        # openai key
        client = OpenAI(
            api_key=api_key
        )
        
        if ('Ovis' in base_model) or ('Gukbap' in base_model):
            average = kllavaw_eval(eval_dataset, model, text_tokenizer, visual_tokenizer, 'ovis', client)
        
        elif 'Bllossom' in base_model:
            average = kllavaw_eval(eval_dataset, model, processor, None, 'bllossom', client)
        
        elif 'VARCO' in base_model:
            average = kllavaw_eval(eval_dataset, model, processor, None, 'VARCO', client)
            
        elif 'gpt' in base_model:
            average = kllavaw_eval(eval_dataset, model, None, None, 'GPT', client)
        
        print("### K-LLABA-W score:", average*10)
    
    else:
        raise Exception("### Not implementation!!")


if __name__ == '__main__':
    torch.cuda.empty_cache() 
    fire.Fire(main)