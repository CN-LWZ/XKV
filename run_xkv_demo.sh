model_path=$1 #Your local model path. such as "mymodel/Meta-Llama-3-8B-Instruct"
total_size=$2 #Total KV cache size for all layers. Such as 4000,8000.
set_value=$3 #Set value for average importance retention ratio.Such as 0.9,0.85.
sampling=$4 #Sampling ratio. Such as 0.1,0.2.

python xkv_demo.py \
    --model_path ${model_path} \
    --total_size ${total_size} \
    --set_value ${set_value} \
    --sampling ${sampling} 