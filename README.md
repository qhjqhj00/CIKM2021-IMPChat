# CIKM2021-IMPChat
CIKM 2021: Learning Implicit User Profile for Personalized Retrieval-based Chatbot ([pdf](https://arxiv.org/pdf/2108.07935.pdf))

In the work, one of the datasets we use is the PchatbotW dataset, please refer to this ([link](https://github.com/qhjqhj00/SIGIR2021-Pchatbot)) for details.


## Datasets
In this paper, we evaluate IMPChat on two datasets, Weibo and Reddit:

### Weibo 
Dataset:
Baidu Disk: [Link](https://pan.baidu.com/s/1NlJPrWqc0VsgDYC_o184aw) (otfo)

Google Storage:[Link](https://drive.google.com/file/d/1_XRL4Uh-izh0pYKq8ZdutJQxtsFwugVl/view?usp=sharing)

Embedding:
Baidu Disk:[Link](https://pan.baidu.com/s/1j_aFGghg6EBYK1HjiDVMUQ) (fob2)

Google Storage:[Link](https://drive.google.com/file/d/1E1-jQu547Qn9lWSOUXTIPJNYcA6VyMd6/view?usp=sharing)

Answer Relevance:
Baidu Disk: [Link](https://pan.baidu.com/s/1SZnk0GLSk6flFZf_Agtbsg) (v6pv)

Google Storage:[Link](https://drive.google.com/file/d/1xKuUnTIdoC2U4LMnLIGU0BJ9-jnin12Z/view?usp=sharing)

### Reddit:
Dataset:
Baidu Disk: [Link](https://pan.baidu.com/s/1OW0P0vfwVd3JSgsGchsIzQ) (ds68)

Google Storage:[Link](https://drive.google.com/file/d/141z3q1yvAgzd81oEKxQ__yPh1kTxrfIK/view?usp=sharing)

Embedding:
Baidu Disk:[Link](https://pan.baidu.com/s/1wzUZV-3FTiiyC6KefWTX-g) (nnie)

Google Storage:[Link](https://drive.google.com/file/d/14RgsYwFuOPCT4-3MtgDh4FvFvXzYNFWT/view?usp=sharing)

Answer Relevance:
Baidu Disk: [Link](https://pan.baidu.com/s/1eBgq6jLwj4vxyNUEjsRxUA) (8mci)

Google Storage:[Link](https://drive.google.com/file/d/1tUTpMGmNhLM9unpebR-_IfSRDX_NnaJ-/view?usp=sharing)

Note that the embeddings are trained on the corresponding dataset. The Answer Relevance file contains the candidate relevances. 

## Train

Download the datasets and put them on the dataset directory.

Run with:

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

## Reproduce the results

Download the checkpoint files and place them under the checkpoint directory:

### Reddit
Baidu Disk: [Link](https://pan.baidu.com/s/1hh8OypwYa7WJeSINL9I9uw) (koon)

Google Storage: [Link](https://drive.google.com/file/d/1b56XHaVaP71ESLaucjiC-p4K0SfKf_UP/view?usp=sharing)

### Weibo
Baidu Disk: [Link](https://pan.baidu.com/s/1E6kRENJ5cRZrJ_pmi_-r4w) (v7ck)

Google Storage: [Link](https://drive.google.com/file/d/10InlyUJaxItEqntymJI3rWVwm-sxqYUj/view?usp=sharing)

Run with:

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
You can download all score files of the baseline models we use in the following links:

Baidu Disk [Link](https://pan.baidu.com/s/1gfdDPv9f3Gs-leRLYQE6NA) (vof6)

Google Storage:[Link](https://drive.google.com/file/d/1A140i9YxMDskRN-Eaad80ZhGNeZc9Hkx/view?usp=sharing)

Each score file is named as {model name}.{task} (e.g. imp.weibo). You can compute the metrics by:

```
python metrics.py
```

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
