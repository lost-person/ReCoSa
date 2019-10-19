# coding = utf-8

"""
hyperparameters of model
"""

import os

data_path = './data/' # data path
res_path = './res/' # results path
ubuntu_data_path = os.path.join('./data/', 'ubuntu_data') # ubuntu data path
ubuntu_res_path = os.path.join('./res/', 'ubuntu_data') # ubuntu data path
jd_data_path = os.path.join('./data/', 'jd') # ubuntu data path
jd_res_path = os.path.join('./res/', 'jd') # ubuntu data path
log_conf = './log.ini' # file path of log config

num_threads = 4 # number of threads 

max_vocab_size = 500000 # maximum number of vocabulary
max_turn = 10 # maximum number of turns
max_uttr_len = 30 # maximum number of words in an utterance

embedding_size = 256 # size of word embedding
hidden_units = 512 # units of hidden layer
num_blocks = 6 # number of encoder/decoder blocks
num_heads = 8 # number of heads in self-attention
drop_rate = 0.1 # dropout rate

num_epochs = 150000 # number of epochs
buffer_size = 1500 # size of buffer
batch_size = 64 # size of batch
eval_step = 10000 # evaluation

lr_rate = 1e-4 # learning rate
