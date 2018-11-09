import torch
import torch.nn as nn
import params 
import numpy as np
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=params.device, dtype=torch.long) * params.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=params.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=params.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class BeamSearch:
    """decoder to generate a sequence of 10 words (nrows) over a vocab of 5 words
    example: 'I like do something that something I like do that'
    """
    def __init__(self):
        self.index2word = {0:'I', 1:'like', 2:'do', 3:'something', 4:'that'}
        data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		    [0.5, 0.4, 0.3, 0.2, 0.1],
		    [0.1, 0.2, 0.3, 0.4, 0.5],
		    [0.5, 0.4, 0.3, 0.2, 0.1],
		    [0.1, 0.2, 0.3, 0.4, 0.5],
		    [0.5, 0.4, 0.3, 0.2, 0.1],
		    [0.1, 0.2, 0.3, 0.4, 0.5],
		    [0.5, 0.4, 0.3, 0.2, 0.1],
		    [0.1, 0.2, 0.3, 0.4, 0.5],
		    [0.5, 0.4, 0.3, 0.2, 0.1]]
        self.data = np.array(data)
    
    
    def greedy(self):
        # for every position out of 10, select the most probably 
        greedy_decoder = [np.argmax(d) for d in self.data]  
        print([self.index2word[d] for d in greedy_decoder])  # ['that', 'I', 'that', 'I', 'that', 'I', 'that', 'I', 'that', 'I']


    def beam(self, k=3):
        """
        Args:
            k: the window size
        """
        sequences = [[list(), 1.0]]
        for d in self.data:
            all_candidates = list()
            # expand each current candidates            
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(d)):
                    candidate = [seq + [j], score*- np.log(d[j])]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:k]
        print(sequences)
        return sequences

if __name__ == '__main__':
    #BeamSearch().greedy()
    BeamSearch().beam()