python solve_turbo.py \
	--dataset_name math \
	--write_mode w \
	--result_path result/turbo-math-3shot.json \
	--num_examplar 3 \
    --demo_path demo/math.json
python evaluate.py --result_path result/turbo-math-3shot.json

python solve_text_002.py \
	--dataset_name math \
	--write_mode w \
	--result_path result/text_002-math-3shot.json \
	--num_examplar 3 \
    --demo_path demo/math.json
python evaluate.py --result_path result/text_002-math-3shot.json

python solve_text_003.py \
	--dataset_name math \
	--write_mode w \
	--result_path result/text_003-math-3shot.json \
	--num_examplar 3 \
    --demo_path demo/math.json
python evaluate.py --result_path result/text_003-math-3shot.json

python solve_claude.py \
	--dataset_name math \
	--write_mode w \
	--result_path result/claude-math-3shot.json \
	--num_examplar 3 \
    --demo_path demo/math.json
python evaluate.py --result_path result/claude-math-3shot.json

python do_math.py \
    --model_name_or_path model/llama \
    --dataset_path dataset/math \
    --output_dir outputs/llama-math \
    --result_path result/llama-math-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 1500 \
    --generation_max_length 2048 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/llama-math-3shot.log
python evaluate.py --result_path result/llama-math-3shot.json

python do_math.py \
    --model_name_or_path model/alpaca \
    --dataset_path dataset/math \
    --output_dir outputs/alpaca-math \
    --result_path result/alpaca-math-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 1500 \
    --generation_max_length 2048 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/alpaca-math-3shot.log
python evaluate.py --result_path result/alpaca-math-3shot.json

python do_math.py \
    --model_name_or_path model/pythia \
    --dataset_path dataset/math \
    --output_dir outputs/pythia-math \
    --result_path result/pythia-math-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 1500 \
    --generation_max_length 2048 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/pythia-math-3shot.log
python evaluate.py --result_path result/pythia-math-3shot.json

python do_math.py \
    --model_name_or_path model/chatglm \
    --dataset_path dataset/math \
    --output_dir outputs/chatglm-math \
    --result_path result/chatglm-math-3shot.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --save_total_limit 1 \
    --save_steps 9999999 \
    --max_source_length 1500 \
    --generation_max_length 2048 \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy epoch \
    --lr_scheduler_type constant \
    --bf16 \
    --report_to none \
> log/chatglm-math-3shot.log
python evaluate.py --result_path result/chatglm-math-3shot.json

python test_falcon_math.py
python evaluate_falcon.py --result_path result/falcon-math-3shot.json