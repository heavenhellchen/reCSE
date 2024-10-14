echo start
cd /home/mnt/zhaofufangchen/SimCSE/
source activate /home/mnt/zhaofufangchen/miniconda3/envs/embedding/
python train.py --model_name_or_path  model/bert-base-uncased --train_file data/nli_for_simcse.csv --output_dir result/sup-bert-same_pos_diff_dropout  --num_train_epochs 3  --per_device_train_batch_size 128  --learning_rate 5e-5  --max_seq_length 32  --evaluation_strategy steps  --metric_for_best_model stsb_spearman  --load_best_model_at_end  --eval_steps 125  --pooler_type cls  --overwrite_output_dir  --temp 0.05  --do_train  --do_eval  --fp16  "$@"
