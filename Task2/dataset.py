import os
import nltk
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Iterator, List, Tuple
from torch.utils.data import Dataset

def split_(sentence: str):
    """
    :param sentence: 要分词的原始句子
    :return:
    """

    snt = sentence.strip()
    snt = snt.lower()
    words = nltk.word_tokenize(snt)

    return words

def read_glove(path):
    """

    :param path: glove词向量的路径
    :return: 读取的glove的词向量和其对应的token字典
    """
    cnt = 0
    word2id = {}
    embedding_list = []
    with open(path, encoding='UTF8', mode='r') as glove_file:
        for line in glove_file:
            if len(line) is not None:
                line = line.rstrip('\n').rstrip()
                line = line.split()
                try:
                    embedding_list.append(np.asarray(line[1:], dtype=np.float).tolist())
                    word2id[line[0]] = cnt
                    cnt += 1
                except:
                    continue
    return embedding_list, word2id

class Data:
    def __init__(self, train_path, val_path, glove_path):
        self.train_path = train_path
        self.val_path = val_path
        self.glove_path = glove_path
        self.train_data = pd.read_csv(train_path, sep='\t')
        self.val_data = pd.read_csv(val_path, sep='\t')

    def get_data(self):
        train_x = self.train_data['Phrase']
        train_y = self.train_data['Sentiment']

        val_x = self.val_data['Phrase']
        val_y = self.val_data['Sentiment']

        return train_x, train_y, val_x, val_y

    def split_sentence(self):
        train_x, train_y, val_x, val_y = self.get_data()
        train_x_list, train_y_list, val_x_list, val_y_list = [], [], [], []

        train_max_len, val_max_len = 0, 0
        for idx, snt in enumerate(train_x):
            snt = split_(snt)
            if len(snt) != 0:
                train_x_list.append(snt)
                train_y_list.append(train_y[idx])
                train_max_len = max(train_max_len, len(snt))

        for idx, snt in enumerate(val_x):
            snt = split_(snt)
            if len(snt) != 0:
                val_x_list.append(snt)
                val_y_list.append(val_y[idx])
                val_max_len = max(val_max_len, len(snt))

        self.max_len = max(train_max_len, val_max_len)

        return train_x_list, train_y_list, val_x_list, val_y_list

    def build_vocab(self):
        print("获取词表。。。")
        train_x, _, val_x, _ = self.split_sentence()
        self.vocab = {}
        self.word_freq = {}
        cnt = 1
        for ins in train_x + val_x:
            for word in ins:
                if word not in self.vocab.keys():
                    self.vocab[word] = cnt
                    cnt += 1
                try:
                    self.word_freq[word] += 1
                except:
                    self.word_freq[word] = 1
        self.vocab_size = len(self.vocab) + 1
        print("词表长度为: ", self.vocab_size)

    def input2tensor(self):
        train_save_path = './data/train_data.pkl'
        val_save_path = './data/val_data.pkl'

        if os.path.exists(train_save_path) and os.path.exists(val_save_path):
            with open(train_save_path, 'rb') as train_file:
                train_data = pickle.load(train_file)
            with open(val_save_path, 'rb') as val_file:
                val_data = pickle.load(val_file)
        else:
            train_x_list, train_y_list, val_x_list, val_y_list = self.split_sentence()
            train_seq_len = [len(seq) for seq in train_x_list]
            train_x_vec = [[self.vocab[word] for word in snt] for snt in train_x_list]
            train_data = (train_x_vec, train_seq_len, train_y_list)

            val_seq_len = [len(seq) for seq in val_x_list]
            val_x_vec = [[self.vocab[word] for word in snt] for snt in val_x_list]
            val_data = (val_x_vec, val_seq_len, val_y_list)

            with open(train_save_path, 'wb') as train_file:
                pickle.dump(train_data, train_file)
            with open(val_save_path, 'wb') as val_file:
                pickle.dump(val_data, val_file)

        return train_data, val_data

    def build_embedding_matrix(self, embed_type, embed_dim):
        print("获取嵌入矩阵。。。")
        embedding_path = './embedding/' + embed_type + '_embedding.pkl'
        if os.path.exists(embedding_path):
            with open(embedding_path, 'rb') as embedding_file:
                embedding_matrix = pickle.load(embedding_file)
        else:
            if embed_type == 'Random':
                embedding_matrix = None
            elif embed_type == 'GloVe':
                cnt = 0
                embedding_list, word2id = read_glove(self.glove_path)
                embedding_matrix = [np.zeros(embed_dim, dtype=np.float).tolist()]
                unk_embedding = np.random.randn(embed_dim).tolist()
                for key in list(self.vocab.keys()):
                    if key in word2id.keys():
                        cnt += 1
                        embedding_matrix.append(embedding_list[word2id[key]])
                    else:
                        embedding_matrix.append(unk_embedding)
                with open(embedding_path, 'wb') as embedding_file:
                    pickle.dump(embedding_matrix, embedding_file)
                print("可用GloVe词向量个数为：", cnt)
            elif embed_type == 'word2vec':
                with open(embedding_path, 'rb') as embedding_file:
                    embedding_matrix = pickle.load(embedding_file)

        return embedding_matrix

class MyDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __getitem__(self, idx):
        vec = self.data[0][idx]
        seq_len = self.data[1][idx]
        label = self.data[2][idx]
        vec = vec + [0] * (self.max_len - seq_len)
        vec = torch.tensor(vec, dtype=torch.long)

        return vec, label, seq_len

    def __len__(self):
        return len(self.data[0])




