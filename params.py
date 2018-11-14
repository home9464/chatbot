import os


device = 'cpu'  # or 'cuda'

MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 2  # Minimum word count threshold for trimming

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

#corpus_name = 'tableau'  # change this to use different datasets
corpus_name = 'math_add'  # change this to use different datasets
#corpus_name = 'movie'  # change this to use different datasets

save_dir = "model"
data_dir = "data"
data_file = os.path.join(data_dir, "{}.tsv".format(corpus_name))

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 5000
print_every = n_iteration // 100
save_every = n_iteration // 5
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
embedding_size = 256
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.5
batch_size = 64
