import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pre_data_w2v import neg_sample

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim, neg_num, dist):
        super(CBOW, self).__init__()
        self.emb_size = vocab_size
        self.emb_dimension = embed_dim
        self.neg_num = neg_num
        self.dist = dist
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)  # 定义输入词的嵌入字典形式
        # self.w_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)  # 定义输出词的嵌入字典形式
        self._init_embedding()  # 初始化

    def _init_embedding(self):
        int_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        # self.w_embeddings.weight.data.uniform_(-0, 0)

    # 用于测试矩阵检验
    def print_for_test(self, pos_u_emb, pos_w_emb, neg_w_emb, s1, s2, s3, n1, n2, n3):
        print('pos_u_emb size:', pos_u_emb.size())
        print('pos_w_emb size:', pos_w_emb.size())
        print('neg_w_emb size:', neg_w_emb.size())
        print('s1 size:', s1.size())
        print('s2 size:', s2.size())
        print('s3 size:', s3.size())
        print('n1 size:', n1.size())
        print('n2 size:', n2.size())
        print('n3 size:', n3.size())

    # 正向传播，输入batch大小得一组（非一个）正采样id，以及对应负采样id
    # pos_u：上下文矩阵, pos_w：中心词矩阵，neg_w：负采样矩阵
    def forward(self, pos_u, pos_w):
        neg_w = neg_sample(self.neg_num, pos_w, self.dist)
        neg_w = torch.from_numpy(neg_w).cuda()
        pos_u_emb = self.u_embeddings(pos_u)   #b * l * d
        pos_w_emb = self.u_embeddings(pos_w).unsqueeze(1)  # b  * 1 * d
        neg_ws_emb = self.u_embeddings(neg_w)  # b * 5 * d

        loss1 = torch.bmm(pos_u_emb, pos_w_emb.transpose(1, 2))
        loss1 = loss1.squeeze()
        loss1 = -F.logsigmoid(loss1.sum())

        loss2 = 0.0
        for i in range(neg_ws_emb.shape[1]):
            neg_w_emb = neg_ws_emb[:, i, :].unsqueeze(1)
            temp = torch.bmm(pos_u_emb, neg_w_emb.transpose(1, 2))
            temp = temp.squeeze()
            loss2 += F.logsigmoid(-temp.sum())

        loss = loss1 - loss2

        return loss


