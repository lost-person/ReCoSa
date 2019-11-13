# coding = utf-8

"""
evaluate model
"""

import os
import pickle
import time
import numpy as np
import tensorflow as tf

from data_helpers import get_record_parser
import hyparams as hp
from model import Model
from train import load_tfrecord
from vocab import get_vocab_size, load_idx2word, trans_idxs2sen
from utils import get_args, save_tfsummary, save_atten_summary, Log
from metrics import save_tgt_pred_sens, cal_bleu, cal_distinct, embed_metrics
from pretrain_models import load_word2vec


def evaluate(test_record_file, vocab_path, word_embed_path, idx2word_path, res_path):
    """
    evaluate model

    Args:
        test_record_file: str path of test tf.record
        vocab_path: str path of vocabulary
        word_embed_path: str path of pretrianed word2vec
        idx2word_path: str path of index2word dictionary path
        res_path: str path of model results
    """

    # get size of vocab and pretrained word embedding
    vocab_size = get_vocab_size(vocab_path)
    if not vocab_size:
        Log.info("get vocab_size err: vocab_path = {}".format(vocab_path))
        return
    
    if not os.path.exists(word_embed_path):
        return
    word_embed = load_word2vec(word_embed_path)

    # load test dataset
    test_dataset = load_tfrecord(test_record_file, FLAGS, False)
    if not test_dataset:
        Log.info("load test dataset err!")
        return
    
    ckpt_path = os.path.join(res_path, 'ckpt')

    pred_path = os.path.join(res_path, 'pred')
    if not os.path.exists(pred_path):
        return

    session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # data iterator
        test_iterator = test_dataset.make_initializable_iterator()
        handle = tf.placeholder(tf.string, shape=[])
        
        # restore model
        model = Model(test_iterator, vocab_size, FLAGS)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        latest_ckpt_path = tf.train.latest_checkpoint(ckpt_path)
        Log.info("load model start: ckpt_path = {}".format(latest_ckpt_path))
        saver = tf.train.Saver()
        saver.restore(sess, latest_ckpt_path)
        Log.info("load model success")

        # test summaries
        test_summary_dir = os.path.join(res_path, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)

        def dev_step():
            """
            evaluate step
            """
            sess.run(test_iterator.initializer)
            test_handle = sess.run(test_iterator.string_handle())
            loss_list = []
            ppl_list = []
            target_list = []
            res_idx_list = []
            pred_idx_list = []
            acc_list = []
            while True:
                try:
                    feed_dict = {
                        handle: test_handle,
                        model.dropout_rate: 0.0
                    }
                    _, loss, ppl, acc, target, res, preds = sess.run([global_step, model.loss, 
                        model.ppl, model.acc, model.target, model.response, model.preds], feed_dict)
                    
                    target_list.extend(target)
                    res_idx_list.extend(res)
                    pred_idx_list.extend(preds)
                    loss_list.append(loss)
                    ppl_list.append(ppl)
                    acc_list.append(acc)
                    
                except tf.errors.OutOfRangeError:
                    break
            
            target_list = [target.decode() for target in target_list]
            idx2word = load_idx2word(idx2word_path)
            pred_list = [trans_idxs2sen(pred_idx_list[i], idx2word).split("</s>", 1)[0].strip() 
                for i in range(len(pred_idx_list))]      
            loss = np.mean(loss_list)
            ppl = np.mean(ppl_list)
            acc = np.mean(acc)
            bleu_score = cal_bleu(target_list, pred_list)
            save_tgt_pred_sens(os.path.join(pred_path, 'test_tgt_pred.txt'), target_list, pred_list)
            dist1 = cal_distinct(pred_list)
            dist2 = cal_distinct(pred_list, 2)
            greedy_match, embed_avg, vec_extrema = embed_metrics(res_idx_list, pred_idx_list, word_embed)
            Log.info("=" * 40)
            Log.info("loss: {:.3f}\t| bleu: {:.3f}\t| ppl: {:.3f}\t| dist_1 = {:.3f}, dist_2 = {:.3f}\t| "
                    "greedy_match = {:.3f}, embed_avg = {:.3f}, vec_extrema = {:.3f}".format(loss, 
                    bleu_score, ppl, dist1, dist2, greedy_match, embed_avg, vec_extrema))
            Log.info("=" * 40)
        
        dev_step()


if __name__ == "__main__":
    FLAGS = get_args()
    Log.info("Parameters:")
    for attr, value in FLAGS.flag_values_dict().items():
        Log.info("{}: {}".format(attr, value))
    
    test_record_file = os.path.join(FLAGS.data_path, 'test.tfrecords')
    vocab_path = os.path.join(FLAGS.data_path, 'vocab.txt')
    word_embed_path = os.path.join(FLAGS.data_path, 'w2v.pkl')
    idx2word_path = os.path.join(FLAGS.data_path, 'idx2word.pkl')
    res_path = os.path.join(FLAGS.res_path, '')
    evaluate(test_record_file, vocab_path, word_embed_path, idx2word_path, res_path)
