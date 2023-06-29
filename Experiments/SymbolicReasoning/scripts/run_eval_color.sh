python solve_turbo.py \
	--dataset_name color \
	--write_mode w \
	--result_path result/turbo-color-3shot.json \
	--num_examplar 3

python solve_text_002.py \
	--dataset_name color \
	--write_mode w \
	--result_path result/text_002-color-3shot.json \
	--num_examplar 3

python solve_text_003.py \
	--dataset_name color \
	--write_mode w \
	--result_path result/text_003-color-3shot.json \
	--num_examplar 3

python solve_claude.py \
	--dataset_name color \
	--write_mode w \
	--result_path result/claude-color-3shot.json \
	--num_examplar 3

python do_color.py \
    --model_name_or_path model/llama \
    --dataset_path dataset/colored_objects \
    --output_dir outputs/llama-color \
    --result_path result/llama-color-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 400 \
    --generation_max_length 410 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/llama-color-3shot.log

python do_color.py \
    --model_name_or_path model/alpaca \
    --dataset_path dataset/colored_objects \
    --output_dir outputs/alpaca-color \
    --result_path result/alpaca-color-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 400 \
    --generation_max_length 410 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/alpaca-color-3shot.log

python do_color.py \
    --model_name_or_path model/pythia \
    --dataset_path dataset/colored_objects \
    --output_dir outputs/pythia-color \
    --result_path result/pythia-color-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 400 \
    --generation_max_length 410 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/pythia-color-3shot.log

python do_color.py \
    --model_name_or_path model/chatglm \
    --dataset_path dataset/colored_objects \
    --output_dir outputs/chatglm-color \
    --result_path result/chatglm-color-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 400 \
    --generation_max_length 410 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/chatglm-color-3shot.log

python test_falcon_color.py