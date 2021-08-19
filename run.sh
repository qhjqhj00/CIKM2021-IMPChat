ts=`date +%Y%m%d%-H%M`
dataset=weibo

CUDA_VISIBLE_DEVICES=4,6 python run.py \
    --task ${dataset} \
    --batch_size 128 \
    --eval_steps 5000 \
    --emb_len 200 \
    --max_utterances 29 \
    --learning_rate 5e-4\
    --max_words 50 \
    --n_gpu 2 \
    --epochs 10 \
    --n_layer 3 \
    --max_hop 2 \
    --score_file_path score_file.txt \
    --model_file_name weibo.MSN.2021-01-31_20:38:43.modify.pt  #${dataset}.IMPChat.pt
    #--is_training False

