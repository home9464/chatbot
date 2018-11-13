import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        """
        Args:
            hidden_size:size of hidden state
            embedding: torch.nn.Embedding, shape=[voc.num_words, hidden_size]

        """
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        Args:
            input_seq: input tensor with shape [max_length, batch_size]
            input_lengths: length of each sentence in the batch, shape=[batch_size]
            hidden: hidden state, shape=[n_layers*num_directions, batch_size, hidden_size]
        Returns:
            outputs: output of last hidden layer (sum of bidirectional outputs),
                     shape=[max_length, batch_size, hidden_size]
            hidden: updated hidden state, shape=[n_layers*num_directions, batch_size, hidden_size]
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)  # shape: [max_length, batch_size, num_features]
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        # hidden: [4, batch_size, feature_size]
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        # outputs: [8, batch_size, feature_size]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : , self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
