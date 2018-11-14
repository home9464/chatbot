import torch
import torch.nn as nn
import params

class EncoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, n_layers=1, dropout=0):
        """
        Args:
            embedding: torch.nn.Embedding, shape=[voc.num_words, embedding_size]
            hidden_size:size of hidden state
        """
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim  # also defined as params.embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size=self.embedding_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          bias=True,
                          batch_first=False,
                          # dropout layer on the outputs of each GRU layer except the last layer
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        GRU: output of shape (seq_len, batch, num_directions * hidden_size):
        tensor containing the output features h_t from the last layer of the GRU,
        for each t (time step).

        If a torch.nn.utils.rnn.PackedSequence has been given as the input,
        the output will also be a packed sequence. For the unpacked case,
        the directions can be separated using
        output.view(seq_len, batch, num_directions, hidden_size),
        with forward and backward being direction 0 and 1 respectively.

        Similarly, the directions can be separated in the packed case.

        h_n of shape (num_layers * num_directions, batch, hidden_size):
        tensor containing the hidden state for t = seq_len

Like output, the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size).

        Args:
            input_seq: input tensor with shape=[max_length, batch_size]
            input_lengths: length of each sentence in the batch, shape=[batch_size]
            hidden: hidden state, shape=[n_layers*num_directions, batch_size, hidden_size]
        Returns:
            outputs: output of last hidden layer (sum of bidirectional outputs),
                     shape=[max_length, batch_size, hidden_size]
            hidden: updated hidden state, shape=[n_layers*num_directions, batch_size, hidden_size]
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)  # shape: [max_length, batch_size, embedding_size]
        # data = [total_valid_words_in_this_batch, embedding_size]
        # batch_sizes = [3,3,2,1], indicate the 1st sentence has 3 words, the 2nd sentence has 3 words, ...
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # outputs: PackedSequence, data.shape=[embedding_size, num_directions*hidden_size],
        # batch_sizes=[64, 64, 64, 64]
        # output of shape (seq_len, batch, num_directions * hidden_size):
        # tensor containing the output features h_t from the last layer of the GRU,
        # for each t (time step).
        # hidden: [num_layers*num_directions, batch_size, hidden_size], tensor containing the hidden state for t = seq_len
        outputs, hidden = self.gru(packed, hidden)

        # outputs: [seq_len, batch_size, num_directions*hidden_size]
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # sum bidirectional GRU outputs, num_directions=0 represents forward, 1 represents backward
        # outputs: [seq_len, batch_size, hidden_size*2] -> [seq_len, batch_size, hidden_size]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # Return outputs and final hidden state when t=seq_len
        # outputs: [seq_len, batch_size, hidden_size]
        # hidden: [num_layers*num_directions, batch_size, hidden_size]
        return outputs, hidden
