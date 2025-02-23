# KoVLMEval
Korean MM Benchmarks Evaluation code
   
# Datasets😎
[K-MMBench](https://huggingface.co/datasets/NCSOFT/K-MMBench).  
[K-MMStar](https://huggingface.co/datasets/NCSOFT/K-MMStar).  
[K=DTCBench](https://huggingface.co/datasets/NCSOFT/K-DTCBench).  
[NCSOFT/K-LLaVA-W](https://huggingface.co/datasets/NCSOFT/K-LLaVA-W).  
> Provided by NCSoft
  
# Download dataset 
```
run.py
dataset (folder)
data (folder)
├──kdtcbench
   └──test-00000-of-00001.parquet
├──kllavaw
   └──test-00000-of-00001.parquet
├──kmmbench
   └──dev-00000-of-00001.parquet
└──kmmstar
   └──val-00000-of-00001.parquet
```
   
# QuickStart🤗
```python
def main(
        dataset = '...dataset.',
        base_model = '...model...',
        cutoff_len = 2048,
        api_key = '...your_api...'
    ):
    
    login(token='...your_token...')
```
> Please set above variables.
  
# Korean VLM Evaluation
| Model | K-MMBench | K-MMStar| K-DTCBench | K-LLAVA-W | Average |
| --- | --- | --- | --- | --- | --- |
| **HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚** | 89.10 | 68.13 | 77.08 | **69.00** | **75.83** |
| **HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚** | 80.16 | 54.20 | 52.92 | 63.83 | 62.78 |
| Ovis2-34B | **89.56** | **68.27** | 76.25 | 53.67 | 71.94 |
| Ovis1.6-Gemma2-9B | 52.46 | 50.40 | 47.08 | 55.67 | 51.40 |
| VARCO-VISION-14B | 87.16 | 58.13 | **85.42** | 51.17 | 70.47 | 
| llama-3.2-Korean-Bllossom-AICA-5B	 | 26.01 | 21.60 | 17.08 | 45.33 | 27.51 |   
> If you want to see our model, Gukbap-VL, please check this [repo🍚](https://github.com/Marker-Inc-Korea/KO-LMM-FFT)!!
    
# Citation
[NCSoft](https://huggingface.co/NCSOFT).  
