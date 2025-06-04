cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-7B-3k-Zero-GRPO-NIG-lora-format-meteor-llm"#Done
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export WANDB_MODE=disabled

IMAGE_ROOT="path/to/route_images_up"
DATASET_NAME="path/to/src/open-r1-multimodal/data_config/nig.yaml"
MODEL_PATH="path/to/Qwen2.5-VL-7B-Instruct"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_nig.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path  $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --image_root $IMAGE_ROOT \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


