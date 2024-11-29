# XKV
XKV: Personalized KV Cache Memory Reduction for Long-Context LLM Inference

## 1. Introduction
Recently the generative Large Language Model (LLM) has achieved remarkable success in numerous applications. Notably its inference generates output tokens one-by-one, leading to many redundant computations. The widely-used KV-Cache framework makes a compromise between time and space complexities. However, caching data generates the increasingly growing memory demand, that can quickly exhaust the limited memory capacity of the modern accelerator like GPUs, particularly in long-context inference tasks. Existing studies reduce memory consumption by evicting some of cached data that have less important impact on inference accuracy. But the benefit in practice is far from ideal due to the static cache allocation across different LLM network layers. 

This paper observes that the layer-specific cached data have very different impacts on accuracy. We quantify this difference, and give experimental and theoretical validation. We accordingly make a formal analysis and shows that customizing the cache size for each layer in a personalized manner can yield a significant memory reduction, while still providing comparable accuracy. We simulate the cache allocation as a combinatorial optimization problem and give a global optimal solution. In particular, we devise a mini- and sampling-based inference over a lightweight variant of the LLM model, so as to quickly capture the difference and then feed it into the personalized algorithms.

### **Key Contributions**
1. **Dynamic Differences of Importance Distribution**
   - Presenting the DDID insight and establishing the equivalence between preserved information in each layer’s cache and the final inference accuracy, with experimental and theoretical validation. 
2. **XKV Framework with Mini-Prefill**:
   - We design a novel inference framework based on miniprefill to support the personalized optimization of KV cache memory.


3. **Personalized & Fast Memory Allocation**:
   - We propose efficient and personalized KV cache allocation and eviction algorithms under two constraint conditions. The algorithms take advantage of DDID to achieve a more efficient reduction of KV cache memory.


4. **Sampling-based Overhead Reduction**:
   - We adopt the sampling strategy to significantly reduce the additional computational overhead introduced, making XKV applicable to multi-scenario tasks.


## 2. Quick Start

Before running it, some softwares must be installed, which is beyond the scope of this document. 
### 2.1 Requirements
This project is run under flash-attn==2.6. Please manually install the corresponding FlashAttention version according to your own CUDA version and PyTorch version.

After that:
```
git clone https://github.com/CN-LWZ/XKV.git
cd XKV
pip install -r requirements.txt
```
### 2.2 Run XKV demo
```
sh run_xkv_demo.sh
```
```
model_path=$1 #Your local model path. such as "mymodel/Meta-Llama-3-8B-Instruct"
total_size=$2 #Total KV cache size for all layers. Such as 4000,8000.
set_value=$3 #Set value for average importance retention ratio.Such as 0.9,0.85.
sampling=$4 #Sampling ratio. Such as 0.1,0.2.

python xkv_demo.py \
    --model_path ${model_path} \
    --total_size ${total_size} \
    --set_value ${set_value} \
    --sampling ${sampling} 
```
- `model_path` is your local model path. Such as "mymodel/Meta-Llama-3-8B-Instruct"
- `total_size` is the first constraint condition, representing total KV cache size of all layers
- `set_value` is the second constraint condition, representing average importance retention ratio
- `sampling` is sampling ratio
- When using one of the constraint conditions, please set the other constraint condition to 0. Both `total_size` and `set_value` cannot be 0 at the same time, nor can they be non-zero at the same time
  
### 2.3 Evaluate XKV inference
```
sh run_xkv_eval.sh
```

## 3. Contact  
If you encounter any problem, please feel free to contact lwzzzz@foxmail.com.
