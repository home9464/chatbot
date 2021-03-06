import os
import csv
import re
import codecs
import random
import itertools
import unicodedata
import torch

from vocab import Voc
import params


def printLines(fileName, n=10):
    with open(fileName, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


def loadConversations(fileName, lines, fields):
    """
    Args:
        fileName: each line has a format like
                  u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
        lines:
            ["lineID", "characterID", "movieID", "character", "text"]
            L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
        fields: ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    Return:
        [
            {
                "character1ID": "L123",
                "character2ID": "L124",
                "movieID": "m3",
                "utteranceIDs": "['L194', 'L195', 'L196', 'L197']"
                "lines":["hello", "world"]
            },
            ...
        ]
    """
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            # u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj["utteranceIDs"])  # ['L194', 'L195', 'L196', 'L197']
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extractSentencePairs(conversations):
    """extract conversation as pairs

    example:
    A: hello
    B: nice to see you
    A: byebye
    B: have a nice day

    ->

    ("hello", "nice to see you")
    (nice to see you", "byebye")
    ("byebye", "have a nice day")
    """
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def normalizeSimpleString(s):
    """apply minimum filter on response(reply, answer)
    because the response is already strictly formatted
    """
    s = unicodeToAscii(s.lower().strip())
    return s

def normalizeMathString(s):
    """
    '12 +   13' -> '1 2 + 1 3'
    '1+15' -> '1 + 1 5'
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"\s+", r"", s).strip()  # remove any spaces
    return ' '.join([_ for _ in s])

def indexesFromSentence(voc, sentence, limiter=' '):
    """convert a sentence into a list of ids
    """
    return [voc.word2index[word] for word in sentence.split(limiter)] + [params.EOS_token]


def zeroPadding(indices, fillvalue=params.PAD_token):
    return list(itertools.zip_longest(*indices, fillvalue=fillvalue))


def binaryMatrix(padlist, value=params.PAD_token):
    """
    Args:
        padlist: list of zero padded word index, example [[1,2,4],[0,3,5],[0,0,6]]

    Returns:
        matrix: example, [[1,1,1], [0,1,1], [0,0,1]]
    """
    m = []
    for i, seq in enumerate(padlist):
        m.append([])
        for token in seq:
            if token == params.PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(sentences, voc):
    """
    Args:
        sentences: ["I", "you do", "he likes that"]
        voc: instance of Vocabulary.Voc

    Returns:
        padVar: tensor[ [1, 2, 4],
                        [0, 3, 5],
                        [0, 0, 6]]
                note it is column-wised for each sentence, so 1st sentence is [1, 0, 0]
                2nd sentence is [2,3,0] and 3rd sentence is [4,5,6]
        lengths: [1, 2, 3]
    """
    #[[indices], [], ...]
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in sentences]
    # tensor([1, 2, 3, ...]), length of each sentence
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(sentences, voc):
    """get padded target sequence tensor, padding mask, and max target length
    Args:
        sentences: ['I very like soccer', "you like food", "here there"]
                    after encoding into id by word2index, above list will be like
                    [[1,2,4,23], [3,4,5], [7,9]]
        voc: instance of Vocabulary.Voc
    Returns:
        padVar: tensor[ [1, 3, 7], [2, 4, 9], [4, 5, 0], [23, 0, 0]]
        mask: tensor[ [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]]
        max_target_len: 4
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in sentences]  # [[1,2,4, 23], [3,4,5], [7,9]]
    max_target_len = max([len(indexes) for indexes in indexes_batch])  # 4, num_rows
    padList = zeroPadding(indexes_batch)  # [ [1,3,7], [2,4,9], [4,5,0], [23, 0, 0]]
    mask = binaryMatrix(padList)# [ [1,1,1], [1,1,1], [1,1,0], [1, 0, 0]]
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    """
    batch_size = number of sentences feeded to the model at each training step
    Args:
        voc: instance of Vocabulary.Voc
        pair_batch: [[pair], [pair]. ,,,]
    Returns:
        inp: input/query with shape of [max_seq_len, batch_size]
        length: length of each sentence in input
        ouput: output/response with shape of [max_seq_len, batch_size]
        mask: mask of output with shape of [max_seq_len, batch_size]
        max_target_len: max length of output sentences
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # padded [max_length, num_words]
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def preprocess(delimiter='\t'):
    """convert raw files into paired conversations like "what is your name<TAB>Tom Hanks"
    and save it as a new file
    """
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(params.corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)

    print("\nLoading conversations...")
    conversations = loadConversations(
        os.path.join(params.corpus, "movie_conversations.txt"),
        lines,
        MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(params.data_file, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)
    #return data_file



def readVocs(datafile, corpus_name):
    """Read query/response pairs and return a voc object
    """
    # each line is a pair of conversation
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    #if corpus_name == 'math_add':
    #    pairs = [[normalizeMathString(s) for s in l.split('\t')] for l in lines]
    #else:
    pairs = [[normalizeSimpleString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < params.MAX_LENGTH and len(p[1].split(' ')) < params.MAX_LENGTH


# Filter pairs using filterPair condition
def filterPairs(pairs):
    """filter long sentences which number of words exceed MAX_LENGTH
    """
    return [pair for pair in pairs if filterPair(pair)]


def loadVocPair(corpus_name, datafile, save_dir):
    """return a populated voc object and pairs list
    """
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def loadPreparedData():
    """load paired conversation and convert into voc
    """
    if not os.path.exists(params.data_file):
        preprocess()
    voc, pairs = loadVocPair(params.corpus_name,
                             params.data_file,
                             params.save_dir)
    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, params.MIN_COUNT)

    #batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(params.BATCH_SIZE)])
    #input_variable, lengths, target_variable, mask, max_target_len = batches
    return voc, pairs


def load_checkpoint(ckpt_filename, vocabulary):
    """torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }
    """
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # If loading on same machine the model was trained on
    checkpoint = torch.load(ckpt_filename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(ckpt_filename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    vocabulary.__dict__ = checkpoint['voc_dict']
    embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')