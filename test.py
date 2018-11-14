import os
import torch
import torch.nn as nn
from torch.autograd import Variable

batch_size = 3
max_length = 3
hidden_size = 2
n_layers =1
num_input_features = 1
vocab_size = 7
embedding_size = 5
#input_tensor = torch.zeros(batch_size, max_length)
#input_tensor[0] = torch.LongTensor([1, 2, 3])
#input_tensor[1] = torch.LongTensor([4, 5, 0])
#input_tensor[2] = torch.LongTensor([6, 0, 0])

input_tensor = torch.LongTensor([[1, 2, 3, 4],
                             [4, 5, 4, 0],
                             [6, 1, 0, 0]])

embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
embedded = embedding(input_tensor)
print(embedded)

seq_lengths = [4, 3, 2]
pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, batch_first=True)
print(pack)
