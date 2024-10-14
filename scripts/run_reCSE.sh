#!/bin/bash

# Train models on wiki1m dataset.

source activate /home/mnt/zhaofufangchen/miniconda3/envs/simcse/

TEXT=/home/mnt/zhaofufangchen/SimCSE/data/wiki1m_for_simcse.txt

SEED=0
MODEL=/home/mnt/zhaofufangchen/SimCSE/model/bert-base-uncased
LR=3e-5
BATCH=64
EPS=3


OUT_DIR=/home/mnt/zhaofufangchen/SimCSE/result/wiki1m/seed_${SEED}/simcse

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# python /home/mnt/zhaofufangchen/SimCSE/simcse/train.py \
#     --framework simcse \
#     --model_name_or_path  $MODEL\
#     --text_file $TEXT \
#     --output_dir $OUT_DIR \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BATCH \
#     --num_train_epochs $EPS  \
#     --seed $SEED

# python /home/mnt/zhaofufangchen/SimCSE/simcse_to_huggingface.py --path $OUT_DIR

python /home/mnt/zhaofufangchen/SimCSE/simcse/evaluation.py \
        --model_name_or_path $OUT_DIR \
        --pooler cls_before_pooler \
        --task_set sts \
        --mode test