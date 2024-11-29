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
```bash

### 2.3 Graph Input Format&&get binary file
We can read the adjacency list and generate the corresponding binary file.
```
src1   dst1:dst2
src2   dst3:dst4:dst5
```
Run
```
../bin/convert2binary input_file_path  dataset_path

```
out: vlist.bin  elist.bin wlist.bin(int) flight.bin(float)
### 2.4 C-2graph-pri
Run
```
../bin/purn-sssp dataset_path

```
### 2.5 C-2graph-P
Run and compare the results
```
../bin/sssp-base dataset_path source
../bin/sssp-purn dataset_path source
```
### 2.6 C-2graph-PM
example
## 3. Contact  
If you encounter any problem with LRCNN, please feel free to contact lihuaibei7951@stu.ouc.edu.cn.
