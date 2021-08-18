# CIKM2021-IMPChat
CIKM 2021: Learning Implicit User Profile for Personalized Retrieval-based Chatbot

In the work, one of the datasets we use is the PchatbotW dataset, please refer to https://github.com/qhjqhj00/SIGIR2021-Pchatbot for details.


### Weibo 
Dataset:
Link: https://pan.baidu.com/s/1NlJPrWqc0VsgDYC_o184aw  
Password: otfo

Embedding:
Link: https://pan.baidu.com/s/1j_aFGghg6EBYK1HjiDVMUQ
Password: fob2

Answer Relevance:
Link: https://pan.baidu.com/s/1SZnk0GLSk6flFZf_Agtbsg  
Password: v6pv


### Reddit:
Dataset:
Link: https://pan.baidu.com/s/1OW0P0vfwVd3JSgsGchsIzQ  
Password: ds68

Embedding:
Link: https://pan.baidu.com/s/1wzUZV-3FTiiyC6KefWTX-g  
Password: nnie

Answer Relevance:
Link: https://pan.baidu.com/s/1eBgq6jLwj4vxyNUEjsRxUA
Password: 8mci

### Train

Download the datasets and put them on the dataset directory.

```
ts=`date +%Y%m%d%-H%M`
dataset=weibo # or reddit

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
    --model_file_name ${dataset}_impchat.pt\
    --is_training True
```

### Reproduce the results

Download the checkpoint files and place them under the checkpoint directory:

Reddit
Link: https://pan.baidu.com/s/1hh8OypwYa7WJeSINL9I9uw  
Password: koon

Weibo
Link:
Password:

```
ts=`date +%Y%m%d%-H%M`
dataset=weibo # or reddit

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
    --model_file_name ${dataset}_impchat.pt
```

### Baseline models

We will provide score files for all baseline models for further study.

### Cite:
```
@inproceedings{qian2021impchat,
     author = {Hongjin Qian and Zhicheng Dou and Yutao Zhu Yueyuan Ma and Ji-Rong Wen}, 
     title = {Learning Implicit User Profile for Personalized Retrieval-based Chatbot}, 
     booktitle = {Proceedings of the {CIKM} 2021}, 
     publisher = {{ACM}}, 
     year = {2021},
     url = {https://doi.org/10.1145/3459637.3482269},
     doi = {10.1145/3459637.3482269}
```

```
@inproceedings{qian2021pchatbot,
     author = {Hongjin Qian and Xiaohe Li and Hanxun Zhong and Yu Guo and Yueyuan Ma and Yutao Zhu and Zhanliang Liu and Zhicheng Dou and Ji-Rong Wen}, 
     title = {Pchatbot: A Large-Scale Dataset for Personalized Chatbot}, 
     booktitle = {Proceedings of the {SIGIR} 2021}, 
     publisher = {{ACM}}, 
     year = {2021}, 
     url = {https://doi.org/10.1145/3404835.3463239}, 
     doi = {10.1145/3404835.3463239}}
```
