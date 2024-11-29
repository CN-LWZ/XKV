import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

def singleton(cls):
    instances={}
    def getinstance(*args,**kwargs):
        if cls not in instances:
            instances[cls]=cls(*args,**kwargs)
        return instances[cls]
    return getinstance

@singleton
class XKV_Var:
    def __init__(self,layers=32,heads=32,lenth=2000) -> None:
        self.each_layer_sum=torch.zeros(layers)
        self.each_layer_distri=torch.zeros((layers,heads,lenth))
        self.num_layer=layers
        self.num_heads=heads
        self.lenth=lenth
        
    def reset(self,q_len=2000):
        
        self.each_layer_distri=torch.zeros((self.num_layer,self.num_heads,q_len))
        self.lenth=q_len
        
    def load(self,atte_weight,layer_idx):
        self.each_layer_sum[layer_idx]=atte_weight.sum()
        attn=torch.sort(atte_weight,dim=-1).values
        attn=attn.view(self.num_heads,atte_weight.shape[-1])
        attn=attn[:,-self.lenth:]   
        self.each_layer_distri[layer_idx]=attn
    
    def return_bestkv_byN(self,total_size):
        
        list_wait=[0]*self.num_layer
        list_length=[0]*self.num_layer
        #begin
        for j in range(self.num_layer):
            temp=self.each_layer_distri[j]
            temp=temp[:,-1:]
            x=temp.sum()/self.each_layer_sum[[j]]
            list_wait[j]=x

        for i in range(total_size):
            #Select the the maximum value of list_wait
            #Find the layer index based on the maximum value
            #Add 1 to the length of the corresponding layer
            #Update the value corresponding to the list_wait index
            max1=max(list_wait)
            layer_id=list_wait.index(max1)
            list_length[layer_id]+=1
            position=list_length[layer_id]
            temp=self.each_layer_distri[layer_id]
            x=temp[:,-(position+1):-position].sum()/self.each_layer_sum[layer_id]
            list_wait[layer_id]=x

        return list_length
    
    def return_bestkv_byRavg(self,set_value):
        
        list_wait=[0]*self.num_layer
        list_length=[0]*self.num_layer
        #begin
        for j in range(self.num_layer):
            temp=self.each_layer_distri[j]
            temp=temp[:,-1:]
            x=temp.sum()/self.each_layer_sum[[j]]
            list_wait[j]=x
        
        R_total=0

        while set_value*self.num_layer>R_total:
            #Select the the maximum value of list_wait
            #Find the layer index based on the maximum value
            #Add 1 to the length of the corresponding layer
            #Update the value corresponding to the list_wait index
            max1=max(list_wait)
            R_total=R_total+max1
            layer_id=list_wait.index(max1)
            list_length[layer_id]+=1
            position=list_length[layer_id]
            temp=self.each_layer_distri[layer_id]
            x=temp[:,-(position+1):-position].sum()/self.each_layer_sum[layer_id]
            list_wait[layer_id]=x

        return list_length
        
class XKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 8, max_capacity_prompt = 128 - 8, kernel_size = 7, pooling = 'avgpool',layer_idx=None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 8, max_capacity_prompt = 128 - 8, kernel_size = 7, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt  > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len-self.window_size <= self.max_capacity_prompt:          
            print(f"{self.layer_idx} layer, KV cache size is {q_len}")
            return key_states, value_states
        
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            
            print(f"{self.layer_idx} layer, KV cache size is {self.max_capacity_prompt+self.window_size}")
            
            indices = attn_cache.topk(self.max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states    
    
    def load_data(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        temp_data=XKV_Var()
        # if(self.layer_idx==0 and attn_cache.shape[-1]<2000):
        #     temp_data.reset(attn_cache.shape[-1])
        # if(self.layer_idx==0 and attn_cache.shape[-1]>=2000):
        #     temp_data.reset()
        if(self.layer_idx==0):
            temp_data.reset(int(attn_cache.shape[-1]))
        temp_data.load(attn_cache,self.layer_idx)
        return
    
    def findsizeR(self,attn_cache,remain_ratio):
        # 根据loss找size
        total=attn_cache.sum()
        for max_capacity_prompt in range(1,attn_cache.shape[-1]):
            compress=attn_cache.topk(max_capacity_prompt, dim=-1).values
            compress=compress.sum()
            if(compress/total>=remain_ratio) :break
        return max_capacity_prompt
           
def init_Xkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 8
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    
    self.xkv_cluster = XKVCluster( 
        num_hidden_layers =self.config.num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size,  
        kernel_size = self.config.kernel_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        pooling = self.config.pooling
        )
    
    
def init_xkvvar(layers,heads,length=2000):
    var=XKV_Var(layers,heads,length)
    return var

def XKV_miniprefill(model,tokenizer,tokenized_prompts,output_max_len,context_length,args,layers):
    for i in range(layers):
        model.model.layers[i].self_attn.config.miniprefill=1
        model.model.layers[i].self_attn.config.kv_seq_len = 0
        
    model.generate( 
        **tokenized_prompts,
        max_new_tokens=output_max_len,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        min_length=context_length+1,
        eos_token_id=[tokenizer.eos_token_id]
    )    
    temp=XKV_Var()
    if(args.total_size==0 and args.set_value!=0):
        allocation_list=temp.return_bestkv_byRavg(args.set_value)
    else:
        if(args.set_value==0 and args.total_size!=0):
            allocation_list=temp.return_bestkv_byN(args.total_size)
        else:
            raise ValueError('set_value and total_size error')
        
    return allocation_list

def XKV_generate(model,tokenizer,tokenized_prompts,output_max_len,context_length,args,layers,allocation_list):
        window_size = 8
        kernel_size = 7
        pooling = "avgpool"
        if not isinstance(window_size, list):
            window_sizes = [window_size] * layers    
        if not isinstance(kernel_size, list):
            kernel_sizes = [kernel_size] * layers
        for i in range(layers):
            model.model.layers[i].self_attn.config.miniprefill=0
            model.model.layers[i].self_attn.config.window_size = window_sizes[i]
            model.model.layers[i].self_attn.config.max_capacity_prompt = allocation_list[i]
            model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
            model.model.layers[i].self_attn.config.pooling = pooling
            model.model.layers[i].self_attn.config.kv_seq_len = 0
            
        output = model.generate( 
            **tokenized_prompts,
            output_attentions = args.output_attentions,
            max_new_tokens=output_max_len,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )
        return output