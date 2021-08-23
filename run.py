import time
import argparse
import pickle
from impchat import IMPChat
import os
import torch
from torch.utils.data import DataLoader
from dataset import DialogueDataset
from trainer import Trainer
import time

task_dic = {
    'reddit':'./dataset/reddit/',
    'weibo': './dataset/weibo/'
}

ts = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime())

## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='weibo',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--emb_len",
                    default=300,
                    type=int)
parser.add_argument("--max_words",
                    default=50,
                    type=int,
                    help="The maximum number of words for each utterance.")
parser.add_argument("--gru_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--learning_rate",
                    default=1e-3,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=5,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")

parser.add_argument("--eval_steps",
                    default=100,
                    type=int,
                    help="evaluation steps")
parser.add_argument("--batch_size",
                    default=32,
                    type=int)
parser.add_argument("--local_rank", 
                    type=int, 
                    default=-1, 
                    help="local_rank for distributed training on gpus")

parser.add_argument("--n_gpu",
                    default=4,
                    type=int)

parser.add_argument("--n_layer",
                    default=3,
                    type=int)

parser.add_argument("--use_cross_matching",
                    default=True,
                    type=bool)

parser.add_argument("--n_filters",
                    default=128,
                    type=int)

parser.add_argument("--max_hop",
                    default=2,
                    type=int)


parser.add_argument("--exact_sigma",
                    default=0.001,
                    type=float)

parser.add_argument("--sigma",
                    default=0.1,
                    type=float)


parser.add_argument("--type_file",
                    default='/home/qhj/dw/weibo/test.type',
                    type=str,
                    help="relevance of answer (to compute ndcg)")


parser.add_argument("--model_file_name",
                    default='weibo.impchat.pt',
                    type=str,
                    help="descrip exp details")

args = parser.parse_args()
if args.is_training:
    args.save_path += args.task + '.' + IMPChat.__name__ + '.'+ts+".pt"
else:
    args.save_path += args.model_file_name
args.type_file = task_dic[args.task] + 'test.type'

args.score_file_path = task_dic[args.task] + args.score_file_path + IMPChat.__name__

print(args)
print("Task: ", args.task)

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train_model():
    path = task_dic[args.task]
    X_train_utterances, X_train_responses, y_train = pickle.load(file=open(path+f"train_{args.task}.pkl", 'rb')) 
    X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path+f"dev_{args.task}.pkl", 'rb')) 
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))

    model = IMPChat(word_embeddings, args=args) 
    train_dataset = DialogueDataset(X_train_utterances, X_train_responses, y_train)
    train_set = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = DialogueDataset(X_dev_utterances, X_dev_responses)
    test_set = DataLoader(test_dataset, batch_size=args.batch_size)

    trainer = Trainer(model, train_set, test_set, y_train, y_dev, args)
    
    trainer.to_train()


def test_model():
    path = task_dic[args.task]
    X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path+f"test_{args.task}.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))

    model = IMPChat(word_embeddings, args=args) 
    model.load_model(args.save_path)
    test_dataset = DialogueDataset(X_dev_utterances, X_dev_responses)
    test_set = DataLoader(test_dataset, batch_size=args.batch_size)
    trainer = Trainer(model, None, test_set, None, y_dev, args)    
    trainer.evaluate(is_test=True)

if __name__ == '__main__':
    start = time.time()
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
    end = time.time()
    print("use time: ", (end-start)/60, " min")




