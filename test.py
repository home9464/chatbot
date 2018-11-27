import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

batch_size = 3
max_length = 3
hidden_size = 2
n_layers =1
num_input_features = 1
vocab_size = 7
embedding_size = 5

class LRM(nn.Module):
    def __init__(self):
        super(LRM, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 in, 1 out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def LogisticRegression():
    model = LRM()
    input_X = torch.Tensor([[1.0, 2.0],
                            [2.0, 3.0],
                            [3.0, 4.0],
                            [4.0, 5.0],
                            [5.0, 6.0]])

    input_y = torch.Tensor([[2.0],
                             [4.0],
                             [6.0],
                             [8.0],
                             [10.0]])
    #loss = nn.MSELoss(reduction='none')
    loss = nn.MSELoss()
    optimization = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(5000):
        pred_y = model.forward(input_X)
        _loss = loss(pred_y, input_y)
        optimization.zero_grad()
        _loss.backward()
        optimization.step()
        if epoch % 500 ==0:
            print('{}: loss {}'.format(epoch, _loss))

    test_x = torch.Tensor([[14.0, 15.0], [16.0, 17.0], [12.0, 13.0]])
    test_y = model.forward(test_x)
    print(test_y)

LogisticRegression()
"""
# create a tensor
input_X = torch.LongTensor([[1, 2, 3, 4],
                            [4, 5, 4, 0],
                            [6, 1, 0, 0]])

input_y = torch.LongTensor([[1, 2, 3, 4],
                            [4, 5, 4, 0],
                            [6, 1, 0, 0]])

weights =

F.linear(input_tensor, )
_, topi = input_tensor.topk(1)  # topi = tensor([[3], [1], [0]])
indices = [[topi[i][0] for i in range(3)]]
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
embedded = embedding(input_tensor)
#print(embedded)

seq_lengths = [4, 3, 2]
pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, batch_first=True)
#print(pack)
#torch.gather()


leaky = nn.LeakyReLU(12)
values = torch.randn(2)
print(values, leaky(values))
"""