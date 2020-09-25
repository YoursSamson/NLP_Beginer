import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.TextLSTM import LSTMNet
from models.TextCNN import CNNModel
from dataset import Data, MyDataset


class Trainer:
    def __init__(self, args):
        self.args = args

        data = Data(args.train_path, args.val_path, args.glove_path)
        data.build_vocab()
        train_data, val_data = data.input2tensor()
        embedding_matrix = data.build_embedding_matrix(args.embed_type, args.embed_dim)
        train_dataset = MyDataset(train_data, data.max_len)
        val_dataset = MyDataset(val_data, data.max_len)

        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model_type == 'CNN':
            self.model = CNNModel(args, data.vocab_size, embedding_matrix).to(args.device)
        else:
            self.model = LSTMNet(args, data.vocab_size, embedding_matrix).to(args.device)

        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

    def train(self):
        self.model.train()
        best_accuary = 0.0
        global_step = 0
        for epoch in range(self.args.epochs):
            num = 0
            total_loss = 0.0
            train_loader = tqdm(self.train_dataloader)
            for batch_X, batch_y, seq_len in train_loader:
                global_step += 1
                batch_X = batch_X.to(self.args.device)
                batch_y = batch_y.to(self.args.device)

                self.optim.zero_grad()

                if self.args.model_type == 'LSTM':
                    pred = self.model(batch_X, seq_len)
                else:
                    pred = self.model(batch_X)
                loss = self.loss_func(pred, batch_y)

                loss.backward()
                total_loss += loss.item()
                num += 1
                self.optim.step()
                train_loader.set_description("train_loss: {:.3f}".format(loss.item()))

            val_accuary = self.validation()
            print("Epoch: {}, Vlidation_accuary: {:.3f}".format(epoch+1, val_accuary))
            best_accuary = val_accuary
            self.model.train()
        res_path = 'result' + self.args.model_type + '.txt'
        with open(res_path, 'a') as res_file:
            res = args.model_type + '_'+ args.embed_type + ': ' + str(best_accuary) + '\n'
            res_file.write(res)

    def cal_accuary(self, pred, label):
        pred_label = torch.argmax(pred, dim=1)
        correct_num = (pred_label == label).sum().item()
        total = pred.shape[0]

        accuary = correct_num / total
        return accuary

    def validation(self):
        self.model.eval()
        accuary = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_X, batch_y, seq_len in self.val_dataloader:
                batch_X = batch_X.to(self.args.device)
                batch_y = batch_y.to(self.args.device)
                if self.args.model_type == 'LSTM':
                    pred = self.model(batch_X, seq_len)
                else:
                    pred = self.model(batch_X)
                accuary += self.cal_accuary(pred, batch_y)
                cnt += 1

        return accuary / cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./data/train_split.tsv', type=str)
    parser.add_argument('--val_path', default='./data/val_split.tsv', type=str)
    parser.add_argument('--glove_path', default='/remote-home/xyliu/pycharm-project/Final/glove/glove.840B.300d.txt', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_size', default=50, type=int)
    parser.add_argument('--model_type', default='CNN', type=str)
    parser.add_argument('--embed_type', default='GloVe', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_fate', default=0.2, type=float)
    parser.add_argument('--C', default=5, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=2020, type=int)

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
    np.random.seed(args.seed)  # Numpy module. 为numpy设置随机种子
    random.seed(args.seed)  # Python random module.

    trainer = Trainer(args)
    trainer.train()