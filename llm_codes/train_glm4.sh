export CUBLAS_WORKSPACE_CONFIG=:4096:8

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_DEVICE_MAX_CONNECTIONS=1

set -o pipefail

pip install imblearn
pip install deepspeed==0.9.3 
pip install accelerate==0.27.2 
pip install transformers==4.39.2 
pip install bitsandbytes==0.41.3 
pip install py-common 
pip install peft==0.10.0 
pip install triton==2.3.0 
pip install trl==0.8.6 
pip install tiktoken==0.6.0 

set -ex

LR=3e-5
NUM_GPUS=8
RUN_NAME=text
BASE_MODEL_PATH="your model path" #你的glm-4-9b-chat模型所在目录地址
PUB_PATH="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
TRAIN_PATH="path of train_author.json" #train_author.json文件路径
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DEEPSPEED_CONFIG=configs/ds_config_zero2.json

mkdir -p $OUTPUT_DIR

entrypoint="finetune_glm4.py"
args="       --train_format input-output \
             --model_type casual \
             --task_type sft \
             --model_name_or_path $BASE_MODEL_PATH \
             --pub_data $PUB_PATH \
             --train_data $TRAIN_PATH \
             --output_dir $OUTPUT_DIR \
             --lora_rank 32 \
             --lora_alpha 64 \
             --lora_dropout 0.05 \
             --max_source_length 4096 \
             --max_target_length 16 \
             --preprocessing_num_workers 1 \
             --per_device_train_batch_size 1 \
             --gradient_accumulation_steps 16 \
             --warmup_ratio 0.02 \
             --weight_decay 0.01 \
             --adam_beta1 0.9 \
             --adam_beta2 0.95 \
             --adam_epsilon 1e-8 \
             --max_grad_norm 0.3 \
             --learning_rate $LR \
             --lr_scheduler_type cosine \
             --num_train_epochs 4 \
             --logging_steps 1 \
             --evaluation_strategy "no" \
             --save_strategy "steps" \
             --save_steps 250 \
             --report_to "none" \
             --seed 1234 \
             --use_lora True \
             --optim adamw_torch_fused \
             --torch_compile True \
             --bf16 \
             --deepspeed $DEEPSPEED_CONFIG"

log_file=${OUTPUT_DIR}/train.log
rm -f "$log_file"

set -o pipefail  # 设置这个选项以确保管道中的任何命令失败都会被捕获

#单机
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --nnode=1 ${entrypoint} ${args} "$@" 2>&1 | tee -a "$log_file"

#多机
# python -m torch.distributed.launch ${entrypoint} ${args} "$@" 2>&1 | tee -a "$log_file"

ls -lh $OUTPUT_DIR