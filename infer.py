# coding = utf-8

"""
evaluate model
"""

import os
import pickle
import time
import numpy as np
import random
import tensorflow as tf

from data_helpers import get_record_parser, load_tfrecord
import hyparams as hp
from model import Model
from vocab import get_vocab_size, load_idx2word, trans_idxs2sen
from utils import get_args, load_word2vec, save_tfsummary, Log
from metrics import save_infer


def infer(test_record_file, vocab_size, idx2word_path, res_path):
    """
    evaluate model

    Args:
        test_record_file: str path of test tf.record
        vocab_size: int size of vocabulary
        idx2word_path: str path of index2word dictionary path
        res_path: str path of model results
    """
    # create path that restore results of model
    if not os.path.exists(res_path):
        Log.info("invalid path: res_path = {}".format(res_path))
        return
    
    ckpt_path = os.path.join(res_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        Log.info("invalid path: ckpt_path = {}".format(ckpt_path))
        return
    
    pred_path = os.path.join(res_path, 'pred')
    if not os.path.exists(pred_path):
        Log.info("invalid path: pred_path = {}".format(pred_path))
        return

    if not vocab_size:
        Log.info("invalid vocab_size: {}".format(vocab_size))
        return

    # load test dataset
    test_dataset = load_tfrecord(test_record_file, FLAGS, False)
    if not test_dataset:
        Log.info("load test dataset err!")
        return
    
    session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # data iterator
        test_iterator = test_dataset.make_one_shot_iterator()
        
        # restore model
        Log.info("build model start!")
        model = Model(test_iterator, vocab_size, FLAGS)
        Log.info("build model success!")
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        latest_ckpt_path = tf.train.latest_checkpoint(ckpt_path)
        Log.info("load model start: ckpt_path = {}".format(latest_ckpt_path))
        saver = tf.train.Saver()
        saver.restore(sess, latest_ckpt_path)
        Log.info("load model success")

        def dev_step():
            """
            evaluate step
            """
            src_sample_list = []
            tgt_sample_list = []
            pred_idx_list = []
            uttr_sample_attn_list = []
            context_sample_attn_list = []
            res_sample_self_attn_list = []
            res_sample_van_attn_list = []

            cnt = 0
            while True:
                try:
                    context, context_len, _, source, target = test_iterator.get_next()
                    context, context_len, source, target = sess.run([context, context_len, source, target])
                    
                    if cnt == 10:
                        break
                    
                    pred = np.ones((FLAGS.batch_size, FLAGS.max_uttr_len), np.int32) * 2
                    
                    feed_dict = {
                        model.context: context,
                        model.context_len: context_len,
                        model.response: pred,
                        model.dropout_rate: 0.0
                    }
                    
                    for j in range(FLAGS.max_uttr_len):
                        _pred = sess.run([model.preds], feed_dict)
                        
                        pred[:, j] = _pred[0][:, j]
                    
                    src_sample_list.extend(source)
                    tgt_sample_list.extend(target)
                    pred_idx_list.extend(pred)

                    cnt += 1
                    
                except tf.errors.OutOfRangeError:
                    break
            
            src_list = [source.decode() for source in src_sample_list]
            tgt_list = [target.decode() for target in tgt_sample_list]
            idx2word = load_idx2word(idx2word_path)
            pred_list = [trans_idxs2sen(pred_idx, idx2word).split("</s>", 1)[0].strip() for pred_idx in pred_idx_list] 
            save_infer(os.path.join(pred_path, 'test_tgt_pred.txt'), src_list, tgt_list, pred_list)
        
        dev_step()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    FLAGS = get_args()
    FLAGS.infer = True
    Log.info("Parameters:")
    for attr, value in FLAGS.flag_values_dict().items():
        Log.info("{}: {}".format(attr, value))
    
    test_record_file = os.path.join(FLAGS.data_path, 'test.tfrecords')
    vocab_path = os.path.join(FLAGS.data_path, 'vocab.txt')
    vocab_size = get_vocab_size(vocab_path)
    idx2word_path = os.path.join(FLAGS.data_path, 'idx2word.pkl')
    res_path = os.path.join(FLAGS.res_path, 'base')
    infer(test_record_file, vocab_size, idx2word_path, res_path)
