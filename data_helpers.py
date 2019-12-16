# coding = utf-8

"""
数据预处理
"""

import os
import pickle

import numpy as np
import tensorflow as tf

import hyparams as hp
from utils import Log
from vocab import load_word2idx, trans_sen2idxs


def pad(context_list, response_list, word2idx, max_turn=hp.max_turn, max_uttr_len=hp.max_uttr_len):
    """
    pad each sequence(context and response) to the same length

    Args:
        context_list: list list of context
        response_list: list list of response
        word2idx: dict word dictionary
        max_turn: int maximum number of turns
        max_uttr_len: int maximum number of words in an utterance
    Returns:
        context_pad_arr: array padded model input
        context_len_arr: array record the length of original sentence(< max_len)
        res_pad_arr: array padded model output
        source_list: list record the original sentence of context
        target_list: list record the original sentence of response
    """
    if not(len(context_list) and len(response_list)):
        Log.info("invalid data: context_size = {:d}, response_size = {:d}".format(len(context_list), len(response_list)))
        return None, None, None, None, None
    
    Log.info("pad start: context_list_size = {:d}, response_list = {:d}".format(len(context_list), len(response_list)))

    context_idx_list = [] # list of idx that corresponds the word in context_list
    res_idx_list = [] # list of idx that corresponds the word in response_list

    source_list = []
    target_list = []

    for context_sen_list, response in zip(context_list, response_list):
        source_list.append("\t".join(context_sen_list))
        target_list.append(response)
        context_sen_idx_list = []
        # truncate idx list
        for context_sen in context_sen_list:
            context_sen_idx_list.append(trans_sen2idxs(context_sen + " </s>", word2idx)[-max_uttr_len:])
        res_sen_idx_list = trans_sen2idxs(response + " </s>", word2idx)[-max_uttr_len:]
        context_idx_list.append(context_sen_idx_list[:max_turn])
        res_idx_list.append(res_sen_idx_list)

    # model input that will be padded [samples, turns, words]
    context_pad_arr = np.zeros([len(res_idx_list), max_turn, max_uttr_len], np.int32) 
    # model output that will be padded [samples, words]
    res_pad_arr = np.zeros([len(res_idx_list), max_uttr_len], np.int32)

    # record the length of original sentence in context [samples, turns]
    context_len_arr = np.zeros([len(res_idx_list), max_turn], np.int32)

    for i, (context_sen_idx_list, res_sen_idx_list) in enumerate(zip(context_idx_list, res_idx_list)):
        for j in range(len(context_sen_idx_list)):
            # pad
            if len(context_sen_idx_list[j]) < max_uttr_len:
                context_pad_arr[i][j] = np.lib.pad(context_sen_idx_list[j], [0, max_uttr_len - len(context_sen_idx_list[j])],
                    'constant', constant_values=(0, 0))
            else:
                context_pad_arr[i][j] = context_sen_idx_list[j][:]
            # record the length
            context_len_arr[i][j] = len(context_sen_idx_list[j])
        # pad
        res_pad_arr[i] = np.lib.pad(res_sen_idx_list, [0, max_uttr_len - len(res_sen_idx_list)], 'constant', constant_values=(0, 0))
    
    Log.info("pad success!")
    return context_pad_arr, context_len_arr, res_pad_arr, source_list, target_list


def load_dataset(data_path):
    """
    load dataset of diglogue

    data format: context\treposonse
    
    Args:
        data_path: str path of dataset
    Returns:
        context_list list list of context
        response_list list list of response
    """

    if not os.path.exists(data_path):
        Log.info("no data file exists: data_path = {}".format(data_path))
        return None

    context_list = []
    response_list = []

    Log.info("load data file start: data_path = {}".format(data_path))
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            eg_list = line.strip().split('\t')
            if len(eg_list) < 2:
                Log.info("invalid data: line = {}".format(line))
                continue

            context = eg_list[:-1]          
            if len(context) > hp.max_turn:
                context = context[:hp.max_turn]
            response = eg_list[-1]

            context_list.append(context)
            response_list.append(response)
    
    Log.info("load data file success!")
    return context_list, response_list


def gen_tf_records(data_path, word2idx_path, records_name, max_turn=hp.max_turn, 
                    max_uttr_turn=hp.max_uttr_len):
    """
    generate tensorflow records

    Args:
        data_path: str the path of data_set
        word2idx_path: str the path of word2idx
        idx2word_path: str the path of idx2word
        record_name: str the name of records
        max_turn: int maximum number of turns
        max_uttr_turn: int maximum number of words in an utterance
    """
    if not (os.path.exists(data_path) and os.path.exists(word2idx_path)):
        Log.info("no file exists: data_path = {}, word2idx_path = {}".format(
            data_path, word2idx_path))
        return None
    
    context_list, response_list = load_dataset(data_path)
    word2idx = load_word2idx(word2idx_path)

    if not (len(context_list) and len(response_list) and len(word2idx)):
        Log.info("load data err: context_list_size = {}, response_list_size = {}, word2idx_size = {}".format(
            len(context_list), len(response_list), len(word2idx)))
        return None

    tf_records = tf.python_io.TFRecordWriter(records_name)

    Log.info("load data to tf records start: tf_records = {}".format(records_name))
    context_pad_arr, context_len_arr, res_pad_arr, source_list, target_list = pad(context_list, response_list, word2idx)
    for context_pad, context_len, res_pad, source, target in zip(context_pad_arr, context_len_arr, 
                                                                res_pad_arr, source_list, target_list):
        features = {
            'context': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_pad.tostring()])),
            'context_len': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_len.tostring()])),
            'response': tf.train.Feature(bytes_list=tf.train.BytesList(value=[res_pad.tostring()])),
            'source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source.encode()])),
            'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target.encode()])),
        }
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        tf_records.write(tf_serialized)
    tf_records.close()
    Log.info("load data to tf records success!")


def get_record_parser(max_turn=hp.max_turn, max_uttr_len=hp.max_uttr_len):
    """
    get the parser to decode record

    Args:
        max_turn: int maximum number of turn
        max_uttr_turn: int maximum number of words in an utterance
    """
    def parse_example(example_proto):
        """
        parse each example in record

        Args:
            example_proto: tensor a single serialized example
        """
        
        record_dict = {
            'context': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'context_len': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'response': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'source': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'target': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }

        example = tf.parse_single_example(example_proto, features=record_dict)
        context = tf.reshape(tf.decode_raw(example["context"], tf.int32), [max_turn, max_uttr_len])
        context_len = tf.reshape(tf.decode_raw(example["context_len"], tf.int32), [max_turn])
        response = tf.reshape(tf.decode_raw(example["response"], tf.int32), [max_uttr_len])
        source = example["source"]
        target = example["target"]
        return context, context_len, response, source, target
    return parse_example


def gen_batch_dataset(record_file, parse_example, buffer_size, batch_size, num_threads, is_training=False):
    """
    generate dataset to train model

    Args:
        record_file: str file path of record
        parse_example: func parse each exampel in record
        buffer_size: int size of buffer
        batch_size: int size of batch
        num_threads: int number of threads
    Returns:
        dataset: TFRecordDataset batch of dataset
    """
    num_threads = tf.constant(num_threads, dtype=tf.int32)
    if is_training:
        dataset = tf.data.TFRecordDataset(record_file).map(parse_example, 
                    num_parallel_calls=num_threads).shuffle(buffer_size).repeat().batch(batch_size)
    else:
        dataset = tf.data.TFRecordDataset(record_file).map(parse_example, 
                    num_parallel_calls=num_threads).repeat(1).batch(batch_size)
    return dataset


def load_tfrecord(record_file, FLAGS, is_training):
    """
    load dataset to train

    Args:
        record_file: str file path of record
        is_training: boolean train dataset or other
        FLAGS: tf.flags 
    Returns:
        dataset
    """
    if not os.path.exists(record_file):
        Log.info("no data file exists: record_file = {}".format(record_file))
        return None

    Log.info("load dataset start: record_file = {}".format(record_file))
    parser = get_record_parser(FLAGS.max_turn, FLAGS.max_uttr_len)
    dataset = gen_batch_dataset(record_file, parser, FLAGS.buffer_size, FLAGS.batch_size, 
                FLAGS.num_threads, is_training)
    Log.info("load dataset success!")
    return dataset


if __name__ == "__main__":
    word2idx_path = os.path.join(hp.data_path, 'word2idx.pkl')
    gen_tf_records(os.path.join(hp.data_path, 'train_data.txt'), word2idx_path, 
                    os.path.join(hp.data_path, 'train.tfrecords'))
    gen_tf_records(os.path.join(hp.data_path, 'valid_data.txt'), word2idx_path, 
                    os.path.join(hp.data_path, 'valid.tfrecords'))
    gen_tf_records(os.path.join(hp.data_path, 'test_data.txt'), word2idx_path, 
                    os.path.join(hp.data_path, 'test.tfrecords'))
