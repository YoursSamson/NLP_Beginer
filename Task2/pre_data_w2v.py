import os
import re
import pickle
import numpy as np
from dataset import Data
from torch.utils.data import Dataset

def get_train_data(vocab, orig_data):
    train_data = []
    for snt in orig_data:
        snt_len = len(snt)
        for i in range(2, snt_len - 2):
            context = [snt[i-2], snt[i-1], snt[i+1], snt[i+2]]
            context = [vocab[word] for word in context]
            target = vocab[snt[i]]
            train_data.append((np.array(context, dtype=np.long), target))

    data_path = './data/word2vec_train.pkl'
    if not os.path.exists(data_path):
        with open(data_path, 'wb') as file:
            pickle.dump(train_data, file)
    else:
        with open(data_path, 'rb') as file:
            train_data = pickle.load(file)
    return train_data

def neg_sample(neg_num, pos, dist):
    neg_w = np.random.choice(len(dist), (len(pos), neg_num), p=dist)
    return neg_w

def load_data(train_path, val_path, glove_path):
    data = Data(train_path, val_path, glove_path)
    train_x_list, _, val_x_list, _ = data.split_sentence()
    data.build_vocab()
    orig_data = train_x_list + val_x_list
    train_data = get_train_data(data.vocab, orig_data)

    print("数据实例个数: {}".format(len(train_data)))

    vocab_size = len(data.vocab) + 1
    print("词表长度为：", vocab_size)

    dist = np.array([v for k, v in data.word_freq.items()])
    dist = np.power(dist, 0.75)
    dist = dist / dist.sum()

    return train_data, data.vocab, vocab_size, dist

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        vec = self.data[idx][0]
        label = self.data[idx][1]

        return vec, label

    def __len__(self):
        return len(self.data)

