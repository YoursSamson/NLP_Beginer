import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(CNNModel, self).__init__()
        self.args = args
        self.batch_size = args.batch_size

        if embedding_matrix is None:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=args.embed_dim, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)

        self.conv1 = nn.Conv2d(1, 32, (4, args.embed_dim), stride=1)
        self.conv2 = nn.Conv2d(32, 64, (4, 1), stride=1)
        self.pool = nn.MaxPool2d((2, 1))
        self.dropout = nn.Dropout(args.dropout_fate)
        self.linear1 = nn.Linear(32*25, 128)
        self.linear2 = nn.Linear(128, args.C)
        self.relu = nn.ReLU()

    def forward(self, wordvec):
        text = self.embed(wordvec)
        # text = self.dropout(text)
        text = text.unsqueeze(1)
        output = self.relu(self.conv1(text))
        output = self.pool(output)
        output = self.linear1(output.reshape(output.shape[0], -1))
        output = self.dropout(output)
        pred = self.linear2(output)
        return pred

