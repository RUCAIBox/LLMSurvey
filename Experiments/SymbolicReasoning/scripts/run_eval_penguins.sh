python solve_turbo.py \
	--dataset_name penguins \
	--write_mode w \
	--result_path result/turbo-penguins-3shot.json \
	--num_examplar 3

python solve_text_002.py \
	--dataset_name penguins \
	--write_mode w \
	--result_path result/text_002-penguins-3shot.json \
	--num_examplar 3

python solve_text_003.py \
	--dataset_name penguins \
	--write_mode w \
	--result_path result/text_003-penguins-3shot.json \
	--num_examplar 3

python solve_claude.py \
	--dataset_name penguins \
	--write_mode w \
	--result_path result/claude-penguins-3shot.json \
	--num_examplar 3

python do_penguins.py \
    --model_name_or_path model/llama \
    --dataset_path dataset/penguins \
    --output_dir outputs/llama-penguins \
    --result_path result/llama-penguins-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 710 \
    --generation_max_length 720 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/llama-penguins-3shot.log

python do_penguins.py \
    --model_name_or_path model/alpaca \
    --dataset_path dataset/penguins \
    --output_dir outputs/alpaca-penguins \
    --result_path result/alpaca-penguins-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 710 \
    --generation_max_length 720 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/alpaca-penguins-3shot.log

python do_penguins.py \
    --model_name_or_path model/pythia \
    --dataset_path dataset/penguins \
    --output_dir outputs/pythia-penguins \
    --result_path result/pythia-penguins-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 710 \
    --generation_max_length 720 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/pythia-penguins-3shot.log

python do_penguins.py \
    --model_name_or_path model/chatglm \
    --dataset_path dataset/penguins \
    --output_dir outputs/chatglm-penguins \
    --result_path result/chatglm-penguins-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 710 \
    --generation_max_length 720 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/chatglm-penguins-3shot.log

python test_falcon_penguins.py