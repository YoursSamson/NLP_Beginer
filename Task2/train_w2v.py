import os
import torch
import pickle
from torch.optim import SGD
from models.word2vec import CBOW
from torch.utils.data import DataLoader
from pre_data_w2v import load_data, MyDataset

epochs = 20
neg_num = 5
embed_dim = 300
learning_rate = 0.001
train_path = './data/train_split.tsv'
val_path = './data/val_split.tsv'
glove_path = '/remote-home/xyliu/pycharm-project/Final/glove/glove.840B.300d.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, vocab, vocab_size, dist = load_data(train_path, val_path, glove_path)
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

model = CBOW(vocab_size, embed_dim, neg_num, dist).to(device)

optim = SGD(model.parameters(), lr=learning_rate)

step = 0
for epoch in range(epochs):
    for context, target in dataloader:
        step += 1
        context = context.to(device)
        target = target.to(device)
        optim.zero_grad()
        loss = model(context, target)
        loss.backward()
        optim.step()

        if step % 500 == 0:
            print("Epoch: {}, Loss: {:.3f}".format(epoch+1, loss.item()))

embedding_matrix = model.u_embeddings.weight.to(torch.device('cpu'))
embedding_matrix = embedding_matrix.data
embedding_matrix = embedding_matrix.numpy()
print("词向量矩阵维度：", embedding_matrix.shape)
embedding_matrix.tolist()

embedding_path = './embedding/w2v_embedding.pkl'
with open(embedding_path, 'wb') as file:
    pickle.dump(embedding_matrix, file)


