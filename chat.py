import os
import torch
import torch.nn as nn
import params

from preprocess import loadPreparedData, indexesFromSentence, normalizeSimpleString, normalizeMathString
from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from search import GreedySearchDecoder


######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=params.MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(params.device)
    lengths = lengths.to(params.device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            if params.corpus_name == 'math_add':
                input_sentence = normalizeMathString(input_sentence)
            else:
                input_sentence = normalizeSimpleString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print('Bot:', "Sorry I do not understand that")


def get_lastest_model_file():
    model_subdir = '{}-{}_{}'.format(params.encoder_n_layers, params.decoder_n_layers, params.hidden_size)
    saved_model_dir = os.path.join(params.save_dir, params.corpus_name, model_subdir)
    tar_files = [os.path.join(saved_model_dir, f) for f in os.listdir(saved_model_dir) if f.endswith('_checkpoint.tar')]
    if tar_files:
        tar_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return tar_files[0]
    else:
        raise Exception("Could not find any models under {}".format(os.path.join(params.save_dir, params.corpus_name)))

loadFilename = get_lastest_model_file()
print('Load model: {} from {}'.format(params.corpus_name, loadFilename))
voc, pairs = loadPreparedData()
# If loading on same machine the model was trained on
checkpoint = torch.load(loadFilename)
# If loading a model trained on GPU to CPU
#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']
embedding = nn.Embedding(voc.num_words, params.embedding_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
encoder = EncoderRNN(embedding, params.hidden_size, params.encoder_n_layers, params.dropout)
decoder = LuongAttnDecoderRNN(params.attn_model,
                              embedding,
                              params.hidden_size,
                              voc.num_words,
                              params.decoder_n_layers,
                              params.dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(params.device)
decoder = decoder.to(params.device)
encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)
print('Bot is ready')
evaluateInput(encoder, decoder, searcher, voc)
