#! /bin/bash

MASTER_ADDR="localhost"
MASTER_PORT=2333
# NNODES=2
NNODES=1
# NODE_RANK=${1-"0"}  # 0 or 1
NODE_RANK=0

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# 根据cuda_visible_devices计算可用的gpu数
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# export WANDB_MODE="disabled"
export WANDB_ENTITY="coldog"
export WANDB_PROJECT="biolinklm"

# model
BASE_PATH="/home/jiang/mainland/biolinklm"
export PYTHONPATH=${BASE_PATH}

method=${1-"standard"}  # Options: "linked", "standard"

## Model/Training
model_name="tinyllama-1.1b-3t"
model_name_or_path="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model_max_length=2048
per_device_batch_size=16
gradient_accumulation_steps=2
learning_rate=1e-4    # 2e-4 - 1M tokens

warmup_ratio=0.03

# seed
seed=42

data_root_dir="${BASE_PATH}/dataset/linklm"

pretrained_save_dir="${BASE_PATH}/checkpoints/${method}/${model_name}/len${model_max_length}/"
pretrained_save_dir+="lr${learning_rate}-bs${per_device_batch_size}-G${gradient_accumulation_steps}-N${GPUS_PER_NODE}-wm${warmup_ratio}/"
pretrained_save_dir+="model/${seed}"
run_name="biolinklm-${method}_${model_name}_len${model_max_length}"

mkdir -p ${pretrained_save_dir}

# Pretrain
echo "===== Pretrain ====="
echo "Seed: ${seed}"
echo "Backbone: ${model_name_or_path}"
echo "Max Length: ${model_max_length}"
echo "Batch Size: ${per_device_batch_size} / Gradient Accumulation: ${gradient_accumulation_steps} / Learning Rate: ${learning_rate}"
echo "Dataset: biolinklm - ${method}"
echo "Save to -> ${pretrained_save_dir}"
echo "===================="

torchrun --nproc_per_node $GPUS_PER_NODE \
         --nnodes $NNODES \
         --node_rank $NODE_RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         "${BASE_PATH}/pretrain/main.py" \
            --stage "train" \
            --train_from_scratch False \
            --resume_from_checkpoint False \
            --run_name ${run_name} \
            --output_dir ${pretrained_save_dir} \
            --gradient_checkpointing True \
            --ddp_timeout 3600 \
            --model_name_or_path ${model_name_or_path} \
            --prepend_bos True \
            --append_eos True \
            --model_max_length ${model_max_length} \
            --optim "adamw_torch" \
            --adam_beta1 0.9 \
            --adam_beta2 0.95 \
            --max_grad_norm 1.0 \
            --learning_rate ${learning_rate} \
            --warmup_ratio ${warmup_ratio} \
            --weight_decay 0.0 \
            --lr_scheduler_type "cosine" \
            --num_cycles 0.39758361765043326 \
            --num_train_epochs 1 \
            --per_device_train_batch_size ${per_device_batch_size} \
            --per_device_eval_batch_size ${per_device_batch_size} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --train_data_path "${data_root_dir}/train.tinyllama_${method}_0" \
            --validation_data_path "${data_root_dir}/valid.jsonl" \
            --test_data_path "${data_root_dir}/test.jsonl" \
            --dataloader_num_workers 32 \
            --dataloader_pin_memory False \
            --evaluation_strategy steps \
            --eval_steps 150 \
            --save_strategy steps \
            --save_steps 750 \
            --save_total_limit 25 \
            --seed ${seed} \
            --deepspeed "${BASE_PATH}/configs/ds_config_alpaca.json" \
            --bf16 True \
            --tf32 True \
            --logging_steps 1
