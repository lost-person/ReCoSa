# coding = utf-8

import logging.config
import os
import pickle

import tensorflow as tf

import hyparams as hp


def get_log(log_conf):
    """
    get the log
    
    Args:
        log_conf: str the config file of log
    Returns:
        logger: logger
    """
    logging.config.fileConfig(log_conf)
    return logging.getLogger()


def trans_sen2idx(seq, word2idx):
    """
    transforms sentence to idx list

    Args:
        seq: str sequence
        word2idx: dict word dictionary
    Returns:
        seq_idx_list: list list of index
    """
    idx_list = [word2idx.get(word, 1) for word in seq.split(' ')]
    return idx_list


def trans_idx2sen(idx_list, idx2word):
    """
    transforms sentence to idx list

    Args:
        idx_list: list list of idx
        idx2word: dict idx2word dictionary
    Returns:
        seq_idx_list: list list of index
    """
    seq = [idx2word.get(idx, '<unk>') for idx in idx_list]
    return seq


def get_vocab_size(vocab_path):
    """
    get the size of vocabulary

    Args:
        vocab_path: str path of vocabulary
    Return:
        vocab_size: int size of vocabulary
    """
    if not os.path.exists(vocab_path):
        Log.info("no data file exists: vocab_path = {}".format(vocab_path))
        return None

    Log.info("get size of vocabulary start: vocab_path = {}".format(vocab_path))
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_size = len(f.readlines())
    
    Log.info("get size of vocabulary success: size = {}".format(vocab_size))
    return vocab_size


def get_args():
    """
    get the arguments of train.py

    Returns:
        FLAGS: tf.flags used in train.py
    """

    tf.flags.DEFINE_string('data_path', hp.data_path, 'Path to dataset(default ubuntu dataset).')
    tf.flags.DEFINE_string('res_path', hp.res_path, 'Path to results')
    tf.flags.DEFINE_boolean('reload_model', False, 'Reload model')
    tf.flags.DEFINE_boolean('is_training', True, 'Must be one of train/eval/decode')
    tf.flags.DEFINE_integer('num_threads', hp.num_threads, 'Number of threads')

    # model parameters
    tf.flags.DEFINE_integer("num_blocks", hp.num_blocks, "Number of stacked blocks")
    tf.flags.DEFINE_integer("num_heads", hp.num_heads, "Number of attention heads")
    tf.flags.DEFINE_integer("max_turn", hp.max_turn, "Max number of turn")
    tf.flags.DEFINE_integer("max_uttr_len", hp.max_uttr_len, "Max number of word") 

    tf.flags.DEFINE_integer("embed_dim", hp.embedding_size, "Dimensionality of embedding")
    tf.flags.DEFINE_integer("hidden_dim", hp.hidden_units, "Dimensionality of rnn")
    tf.flags.DEFINE_float("dropout_rate", hp.drop_rate, "Dropout keep probability") 

    # Training parameters
    tf.flags.DEFINE_integer("buffer_size", hp.buffer_size, "Buffer Size")
    tf.flags.DEFINE_integer("batch_size", hp.batch_size, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", hp.num_epochs, "Number of training epochs")
    tf.flags.DEFINE_integer("print_step", hp.print_step, "Print results of model")
    tf.flags.DEFINE_integer("eval_step", hp.eval_step, "Evaluate model on dev set")

    # loss func parameters
    tf.flags.DEFINE_string('optimizer', 'adam', 'Which optimization method to use') # adam 0.001  adadelta
    tf.flags.DEFINE_float('lr_rate', hp.lr_rate, 'learning rate')

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    return FLAGS

Log = get_log(hp.log_conf)
