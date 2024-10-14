echo start
cd /home/mnt/zhaofufangchen/SimCSE/
source activate /home/mnt/zhaofufangchen/miniconda3/envs/embedding/
#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

loss_mode="adnce"
w1="0.4"
w2="1.0"
temp="0.07"
model_name="bert-base-uncased"
model_name_or_path="/home/mnt/zhaofufangchen/SimCSE/model/bert-base-uncased"
per_device_train_batch_size="64"
learning_rate="3e-5"
echo $model_name
output_dir="result/${model_name}_loss_${loss_model}_bsz_${bsz}_lr_${lr}_t_${temp}_w1_${w1}_w2_${w2}"
python /home/mnt/zhaofufangchen/SimCSE/train.py \
    --model_name_or_path $model_name_or_path \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir $output_dir \
    --loss_mode $loss_mode \
    --w1 $w1 \
    --w2 $w2 \
    --temp $temp \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --learning_rate $learning_rate \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --fp16 
