import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.utils.rnn as rnn_utils

from typing import List, Optional

from torch.nn.utils.rnn import pack_padded_sequence


class LSTMNet(nn.Module):
    def __init__(self, args, vocab_size, embedding_matrix, bidirectional=False):
        super(LSTMNet, self).__init__()

        self.args = args
        self.rnn = nn.LSTM(input_size=args.embed_dim, hidden_size=args.hidden_size, num_layers=1,
                           batch_first=True, dropout=0., bidirectional=bidirectional)

        if embedding_matrix is None:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=args.embed_dim, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)

        if bidirectional:
            self.linear = nn.Linear(2 * args.hidden_size, args.C)
        else:
            self.linear = nn.Linear(args.hidden_size, args.C)


    def forward(self, x, seq_len, only_use_last_state=False):
        x_embed = self.embed(x)

        packed_input = rnn_utils.pack_padded_sequence(x_embed, seq_len, batch_first=True, enforce_sorted=False)

        packed_out, (h_n, c_n) = self.rnn(packed_input, None)
        out_rnn = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

        output = out_rnn[0]

        if only_use_last_state:
            output = h_n.squeeze()
        else:
            output = torch.mean(output, dim=1)

        output = self.linear(output)
        return output


