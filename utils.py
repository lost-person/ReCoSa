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


def get_args():
    """
    get the arguments of train.py

    Returns:
        FLAGS: tf.flags used in train.py
    """

    tf.flags.DEFINE_string('data_path', hp.data_path, 'Path to dataset(default ubuntu dataset).')
    tf.flags.DEFINE_string('res_path', hp.res_path, 'Path to results')
    tf.flags.DEFINE_boolean('infer', False, 'Reload model')
    tf.flags.DEFINE_boolean('reload_model', True, 'Reload model')
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


def load_word2vec(w2v_path):
    """
    load word embedding(after look-up)

    Args:
        w2v_path: str file path of word embedding
    Returns:
        pre_word_vec: Word2Vec pretrianed word embedding
    """
    if not os.path.exists(w2v_path):
        Log.info("no data file exists: word2vec_path = {}".format(w2v_path))
        return None
    
    Log.info("load word embedding start: word2vec_path = {}".format(w2v_path))
    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)
    Log.info("load word embedding success!")
    return w2v


def save_tfsummary(writer, step, key, value):
    """
    save summary of tensorflow

    Args:
        writer: FileWriter 
        step: int step
        key: str name of summary
        value: object value of summary
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, step)


def save_atten_summary(summary_writer, step, uttr_attn_list, context_attn_list, dec_self_attn_list, dec_van_attn_list):
    for i, (uttr_layer_attn, context_layer_attn, dec_layer_self_attn, dec_layer_van_attn) in enumerate(zip(
        uttr_attn_list, context_attn_list, dec_self_attn_list, dec_van_attn_list)):
        for j, (uttr_head, context_head, dec_self_head, dec_van_head) in enumerate(zip(uttr_layer_attn, 
            context_layer_attn, dec_layer_self_attn, dec_layer_van_attn)):
            uttr_head_summary = tf.summary.image('uttr_layer{}_head{}'.format(i, j), tf.expand_dims(
                tf.transpose(uttr_head, [0, 2, 1]), -1))
            context_head_summary = tf.summary.image('context_layer{}_head{}'.format(i, j), tf.expand_dims(
                tf.transpose(context_head, [0, 2, 1]), -1))
            dec_self_head_summary = tf.summary.image('dec_layer{}_self_head{}'.format(i, j), tf.expand_dims(
                tf.transpose(dec_self_head, [0, 2, 1])[:1], -1))
            dec_van_head_summary = tf.summary.image('dec_layer{}_van_head{}'.format(i, j), tf.expand_dims(
                tf.transpose(dec_van_head, [0, 2, 1]), -1))
            summary_writer.add_summary(uttr_head_summary.eval(), step)
            summary_writer.add_summary(context_head_summary.eval(), step)
            summary_writer.add_summary(dec_self_head_summary.eval(), step)
            summary_writer.add_summary(dec_van_head_summary.eval(), step)


Log = get_log(hp.log_conf)
