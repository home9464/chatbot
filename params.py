import os


device = 'cpu'
#device = 'cuda'
MAX_LENGTH = 20  # Maximum sentence length to consider
#MAX_SENTENCE_LENGTH = 30  # Maximum sentence length (number of words) to consider

MIN_COUNT = 2  # Minimum word count threshold for trimming
BATCH_SIZE = 64

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


corpus_name = 'movie-dialogs'
corpus = os.path.join('data', corpus_name)
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = n_iteration // 5

save_dir = os.path.join("data", "save")
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 1000
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.5
batch_size = 32
