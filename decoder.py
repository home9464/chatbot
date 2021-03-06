import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attn


class LuongAttnDecoderRNN(nn.Module):
    """feed batch one time step at a time.
    """
    def __init__(self,
                 attn_model,
                 embedding,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 dropout=0.1):
        """
        Args:
            attn_model: 'dot' or 'general' or ''
            embedding: embedding of current input word, shape=[1, batch_size, embedding_size]
            hidden_size: int
            output_size: vocabulary.num_words
        """
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        # applies a linear transformation to the incoming data: y=xAT+b
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        Args:
            input_step: one word from input sequence with shape=[1, batch_size]
            last_hidden: last hidden layer of GRU
        Retunrs:
            output: output with shape=[batch_size, voc.num_words]
            hidden: shape=[n_layers* num_directions, batch_size, hidden_size]

        """
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)  # [1, batch_size, embedding_size]
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        # rnn_output: [1, batch_size, hidden_size], hidden: [2, batch_size, hidden_size]
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        #encoder_outputs: [max_sent_len, batch_size, hidden_size]
        # attn_weights: [batch_size, 1, max_sent_length]
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch_size, 1, hidden_size]
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)  # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        context = context.squeeze(1)  # [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
