model_path='your model path'
model_name='diversity-7b'
num_gpus=1
cuda_visible_devices=0

# MMLU testing
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python3 mmlu/mmlu.py main --model_name llama --model_path ${model_path} --ntrain 0

# BBH testing 
# log and result will be saved to bbh/logs/your_log_file
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python bbh/src/main.py -b bbh/config/benchmark/bbh10k.yaml -m bbh/config/model/llama.yaml -l bbh/logs/${model_name}.log -p ${model_path} \
model.model_alias llama-model \
model.url http://localhost:5000


# auto-eval testing
# generating response
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} torchrun --master_port=7126 auto_eval/generate.py \
    --base_model ${model_path} \
    --output_file auto_eval/generates/${model_name}.json \
    --input_file 'auto_eval/eval_gpt-3.5-turbo-0301.json' \
    --model_name ${model_name}

# automatic comparison
python auto_eval/eval.py --baseline 'auto_eval/generates/instruct-7b.json' --target 'auto_eval/generates/diversity-7b.json'