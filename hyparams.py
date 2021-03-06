# coding = utf-8

import os

data_name = 'ubuntu_v2'
data_path = os.path.join('data', data_name)
res_path = os.path.join('res', data_name)
log_conf = 'log.ini' # file path of log config

num_threads = 4 # number of threads 

max_turn = 10 # maximum number of turns
max_uttr_len = 20 # maximum number of words in an utterance

embedding_size = 256 # size of word embedding
hidden_units = 512 # units of hidden layer
num_blocks = 6 # number of encoder/decoder blocks
num_heads = 8 # number of heads in self-attention
drop_rate = 0.1 # dropout rate

num_epochs = 500000 # number of epochs
buffer_size = 1500 # size of buffer
batch_size = 64 # size of batch

# print and evaluation step
if data_name.find('cornell') != -1:
    print_step = 100
    eval_step = 1000
else:
    eval_step = 10000
    print_step = 1000

lr_rate = 1e-4 # learning rate
