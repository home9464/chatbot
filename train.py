import os
import random
import torch
import torch.nn as nn
from torch import optim

from preprocess import loadPreparedData, batch2TrainData, indexesFromSentence, normalizeString

from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from search import GreedySearchDecoder

import params


def maskNLLLoss(inp, target, mask):
    """to calculate loss correctly, mask is needed to count number of valid tokens
    e.g. [12, 23, 14, 0, 0] when batch_size = 5
    the mask will be [1, 1, 1, 0, 0] which means only the 3 valid losses,
    so the total loss will be calculated with 3, not 5

    Args:
        inp: input (predicted result) with shape=[batch_size, vocab_size]
        target: the true word with shape=[batch_size]
        mask: the mask with shape=[batch_size]. e.g [1,1,1,0,0, ...]
    Returns:
        loss: the loss value, shape=[1]
        nTotal: number of valid words in the inp
    """
    nTotal = mask.sum()  # number of words
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    # get the mean with masked results
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(params.device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable,
          mask, max_target_len, encoder, decoder,
          embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip,
          max_length=params.MAX_LENGTH):
    """
    Args:
        input_variable: tensor with shape of [max_sentence_len1, batch_size]
        lengths: length of each sentence in input_variable, [1, 23, 14, ...]
        target_variable:  tensor with shape of [max_sentence_len2, batch_size]
        mask: tensor with shape of  [max_sentence_len2, num_sentences2]
        max_target_len: max length of target sequence in target_variable
        encoder: instance of encoder.Encoder
        decoder: instance of decoder.Decoder
        embedding: instane of torch.nn.Embedding

    """
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(params.device)
    lengths = lengths.to(params.device)
    target_variable = target_variable.to(params.device)
    mask = mask.to(params.device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # encoder_outputs [seq_len, batch_size, hidden_size],
    # encoder_hidden: [num_layers*num_directions , batch_size, hidden_size]
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[params.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(params.device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < params.teacher_forcing_ratio else False
    use_teacher_forcing = False
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )  # decoder_output: [batch_size, vocab_size]
            # No teacher forcing: next input is decoder's own current output
            # for each batch, the index of highest probability
            _, topi = decoder_output.topk(1)
            # select the best word as input for generating next word
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(params.device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# Training iterations

#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#

def trainIters(voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
               save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, load_filename):
    """
    Args:
        voc: instance of vocabulary.Voc
        pairs: [["A","B"], ["C", "D"], ...]
        encoder: instance of encoder.EncoderRNN
        decoder: instance of decoder.LuongAttnDecoderRNN
        encoder_optimizer: torch.optim.Adam
        decoder_optimizer: torch.optim.Adam
        embedding: torch.nn.Embedding
        encoder_n_layers: number of layers for encoder
        decoder_n_layers: number of layers for decoder
        save_dir: the full directory to save model onto
        n_iteration: int, how many iteration to shuffle and sample(batch_size) to generate train data
        batch_size: int
        print_every: int, how frequent to print message at specified interval
        save_every: int, how frequent to save model as .tar file
        clip: clip gradients
        corpus_name:
        load_filename: saved model name

    """
    # Load batches for each iteration
    #training_batches = []
    #for _ in range(n_iteration):
    #    random.shuffle(pairs)
    #    num_total = len(pairs)
    #    for i in range(0, num_total, batch_size):
    #        training_batches.append(batch2TrainData(voc, pairs[i:i+batch_size]))

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if load_filename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
    #for iteration, training_batch in enumerate(training_batches):
    #    print(iteration)
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, params.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def main():
    """primary entry
    """
    voc, pairs = loadPreparedData()
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    #embedding = nn.Embedding(voc.num_words, params.hidden_size)
    embedding = nn.Embedding(voc.num_words, params.embedding_size)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(embedding,
                         params.hidden_size,
                         params.encoder_n_layers,
                         params.dropout)
    decoder = LuongAttnDecoderRNN(params.attn_model,
                                  embedding,
                                  params.hidden_size,
                                  voc.num_words,
                                  params.decoder_n_layers,
                                  params.dropout)
    # Use appropriate device
    encoder = encoder.to(params.device)
    decoder = decoder.to(params.device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=params.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(),
                               lr=params.learning_rate * params.decoder_learning_ratio)

    # Run training iterations
    print("Starting Training!")
    trainIters(voc, pairs, encoder, decoder,
               encoder_optimizer, decoder_optimizer,
               embedding,
               params.encoder_n_layers,
               params.decoder_n_layers,
               params.save_dir,
               params.n_iteration, params.batch_size,
               params.print_every, params.save_every,
               params.clip, params.corpus_name, load_filename=None)

if __name__ == '__main__':
    main()