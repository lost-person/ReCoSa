# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs

def load_de_vocab():
    '''
    源序列的词表，逆词表
    '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    '''
    目标序列的词表，逆词表
    '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents):
    '''
    构建数据集
    '''
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # 源序列和目标序列的索引
    x_list, y_list = [], []

    # 源序列和目标序列
    Sources, Targets = [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        source_sent_split = source_sent.split(u"</d>")
        x = []
        for sss in source_sent_split:
            if len(sss.split())==0:
                continue
            x.append([de2idx.get(word, 1) for word in (sss + u" </S>").split()]) # 1: OOV, </S>: End of Text
        target_sent_split = target_sent.split(u"</d>")
        y = [en2idx.get(word, 1) for word in (target_sent_split[0] + u" </S>").split()] 
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad
    # X, Y 不同是因为源序列和目标序列的数据格式不同
    X = np.zeros([len(x_list), hp.max_turn, hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)

    # 记录源序列的原始长度
    X_length = np.zeros([len(x_list), hp.max_turn], np.int32)
    
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        for j in range(len(x)):
            if j >= hp.max_turn:
                break
            # 填充 0
            if len(x[j]) < hp.maxlen:
                X[i][j] = np.lib.pad(x[j], [0, hp.maxlen-len(x[j])], 'constant', constant_values=(0, 0))
            else:
                X[i][j] = x[j][:hp.maxlen]
            X_length[i][j] = len(x[j])
        #X[i] = X[i][:len(x)]
        #X_length[i] = X_length[i][:len(x)]
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, X_length, Y, Sources, Targets

def load_train_data():
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line]
    
    X, X_length, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, X_length, Y, Sources, Targets
    
def load_test_data():
    def _refine(line):
        #line = regex.sub("<[^>]+>", "", line)
        #line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]
        
    X, X_length, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X,X_length, Y, Sources, Targets # (1064, 150)

def load_dev_data():
    def _refine(line):
        #line = regex.sub("<[^>]+>", "", line)
        #line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_dev, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_dev, 'r', 'utf-8').read().split("\n") if line]
        
    X, X_length, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, X_length,Y, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, X_length, Y, sources,targets = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    X_length = tf.convert_to_tensor(X_length, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, X_length, Y, sources, targets])
            
    # create batch queues
    x, x_length, y,sources, targets = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, x_length, y, num_batch, sources, targets# (N, T), (N, T), ()


def gen_tf_records(records_name, mode='train'):
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
    tf_records = tf.python_io.TFRecordWriter(records_name)

    if mode == 'train':
        X, X_length, Y, Sources, Targets = load_train_data()
    elif mode == 'dev':
        X, X_length, Y, Sources, Targets = load_dev_data()
    else:
        X, X_length, Y, Sources, Targets = load_test_data()

    for x, x_len, y, source, target in zip(X, X_length, Y, Sources, Targets):
        features = {
            'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tostring()])),
            'x_len': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_len.tostring()])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tostring()])),
            'source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source.encode()])),
            'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target.encode()]))
        }
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        tf_records.write(tf_serialized)
    tf_records.close()


def get_record_parser(max_turn=hp.max_turn, max_uttr_len=hp.maxlen):
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
            'x': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'x_len': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'y': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'source': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'target': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }

        example = tf.parse_single_example(example_proto, features=record_dict)
        context = tf.reshape(tf.decode_raw(example["x"], tf.int32), [max_turn, max_uttr_len])
        context_len = tf.reshape(tf.decode_raw(example["x_len"], tf.int32), [max_turn])
        response = tf.reshape(tf.decode_raw(example["y"], tf.int32), [max_uttr_len])
        source = tf.decode_raw(example["source"], tf.string)
        target = tf.decode_raw(example["target"], tf.string)
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


if __name__ == "__main__":
    gen_tf_records('./corpora/train.tfrecords', 'train')
    # gen_tf_records('./corpora/dev.tfrecords', 'dev')
    gen_tf_records('./corpora/test.tfrecords', 'test')
