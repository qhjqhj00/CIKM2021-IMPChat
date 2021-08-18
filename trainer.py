import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import os
import sys
import logging
import time

from metrics import Metrics
from tqdm import tqdm

log = logging.getLogger("impchat")

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def init_log(log, args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    ts = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime())
    output_file = f'log/{ts}_{args.task}_{args.max_utterances}_{args.max_words}_{args.batch_size}_impchat.log'
    fh = logging.FileHandler(output_file, mode='a')
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model

class Trainer:

    def __init__(self, model, train_set, test_set, y_train, y_dev, args):
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(args.score_file_path, args.type_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        init_log(log, args)
        log.info(str(model))
        self.train_set = train_set
        self.test_set = test_set
        self.y_dev = y_dev
        self.y_train = y_train
        self.args = args
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2_reg)

    def train_step(self, i, data):
        with torch.no_grad():
            batch_u, batch_r, batch_y = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()
        logits = self.model(batch_u, batch_r) #, logits2
        loss = self.loss_func(logits, target=batch_y)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(), batch_y.size(0)) )  # , accuracy, corrects
        return loss


    def to_train(self):
        log.info('Start to train...')
        log.info(f'train set: {len(self.y_train)}')
        log.info(f'test set: {len(self.y_dev)}')

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch+1, "/", self.args.epochs)
            avg_loss = 0

            self.model.train()
            for i, data in enumerate(self.train_set):
                loss = self.train_step(i, data)

                if i > 0 and i % self.args.eval_steps == 0:
                    log.info(f'epoch {epoch+1}:')
                    self.evaluate()
                    self.model.train()

                if epoch >= 2 and self.patience >= 3:
                    self.reload()
                    
                if self.patience == -1:
                    self.reload()

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.model.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(self.y_train) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss/cnt))
            self.evaluate()


    def reload(self):
        log.info("Reload the best model...")
                    
        if self.args.n_gpu > 1:
            self.model.module.load_state_dict(torch.load(self.args.save_path))
        else:
            self.model.load_state_dict(torch.load(self.args.save_path))
        self.adjust_learning_rate()
        log.info('lr to:' + str(self.args.learning_rate))
        self.patience = 0

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)


    def evaluate(self, is_test=False):
        y_pred = self.predict()
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, self.y_dev):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )

        result = self.metrics.evaluate_all_metrics()
        log.info(f"Evaluation Result: \n " \
        f"MAP:{result[0]}\tMRR:{result[1]}\t" \
        f"P@1:{result[2]}\tR1:{result[3]}\t" \
        f"R2:{result[4]}\tR5:{result[5]}"\
        f"ndcg: {result[6]}")

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + self.best_result[5]:
            log.info(f"Best Result: \n " \
            f"MAP:{self.best_result[0]}\tMRR:{self.best_result[1]}\t" \
            f"P@1:{self.best_result[2]}\tR1:{self.best_result[3]}\t" \
            f"R2:{self.best_result[4]}\tR5:{self.best_result[5]}"\
            f"ndcg: {result[6]}")
            self.patience = 0
            self.best_result = result
            torch.save(get_model_obj(self.model).state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1

        if not is_test and result[3] + result[4] + result[5] < (self.best_result[3] + self.best_result[4] + self.best_result[5]) * 0.9:
            self.patience = -1


    def predict(self):
        self.model.eval()
        y_pred = []

        for i, data in tqdm(enumerate(self.test_set)):
            with torch.no_grad():
                batch_u, batch_r = (item.cuda() for item in data)
                logits1 = self.model(batch_u, batch_r) # , logits2
                logits = logits1 #  + logits2
                y_pred += logits.data.cpu().numpy().tolist()
        return y_pred




