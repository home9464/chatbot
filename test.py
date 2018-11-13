import os
import torch
import torch.nn as nn
import params
print(torch.nn.Embedding(100, 256))
model_subdir = '{}-{}_{}'.format(params.encoder_n_layers, params.decoder_n_layers, params.hidden_size)
saved_model_dir = os.path.join(params.save_dir, params.corpus_name, model_subdir)
tar_files = [os.path.join(saved_model_dir, f) for f in os.listdir(saved_model_dir) if f.endswith('_checkpoint.tar')]
tar_files.sort(key=lambda x: os.path.getmtime(x))
print(tar_files)

#                            ,
#                            '{}_checkpoint.tar'.format(checkpoint_iter))
