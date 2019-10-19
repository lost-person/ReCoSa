# coding = utf-8

"""
evaluate model
"""

import os
import pickle
import numpy as np
import tensorflow as tf

from my_data_helpers import get_record_parser, get_vocab_size, load_word2vec
import my_hyparams as hp
from my_model import Model
from my_train import load_tfrecord
from my_utils import get_args, trans_idx2sen, save_tgt_pred_sens, cal_bleu, Log


def evaluate(test_record_file, vocab_path, pre_word2vec_path, idx2word_path, res_path):
    """
    evaluate model

    Args:
        test_record_file: str path of test tf.record
        vocab_path: str path of vocabulary
        pre_word2vec_path: str path of pretrianed word2vec
        idx2word_path: str path of index2word dictionary path
        res_path: str path of model results
    """

    # get size of vocab and pretrained word embedding
    vocab_size = get_vocab_size(vocab_path)
    if not vocab_size:
        Log.info("get vocab_size err: vocab_path = {}".format(vocab_path))
        return
    
    if FLAGS.use_pre_wordvec:
        pre_word2vec = load_word2vec(pre_word2vec_path)
    else:
        pre_word2vec = None

    # load test dataset
    test_dataset = load_tfrecord(test_record_file, FLAGS, False)
    if not test_dataset:
        Log.info("load test dataset err!")
        return
    
    ckpt_path = os.path.join(res_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        return
    
    # find path of best model
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    best_ckpt_path = ''
    for all_model_ckpt_path in ckpt.all_model_checkpoint_paths:
        if all_model_ckpt_path.find('best') != 0:
            best_ckpt_path = all_model_ckpt_path
            break
    
    if not best_ckpt_path:
        return

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
        model = Model(test_iterator, vocab_size, pre_word2vec, FLAGS)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        saver = tf.train.Saver()
        Log.info("load model start: best_ckpt_path = {}".format(best_ckpt_path))
        saver.restore(sess, best_ckpt_path)
        Log.info("load model success")

        def dev_step():
            """
            evaluate step
            """
            sess.run(test_iterator.initializer)
            test_handle = sess.run(test_iterator.string_handle())
            target_list = []
            pred_idx_list = []

            while True:
                try:
                    feed_dict = {
                        handle: test_handle,
                        model.dropout_rate: 0.0
                    }
                    step, target, preds = sess.run([global_step, model.target, model.preds], feed_dict)
                    
                    target_list.extend(target)
                    pred_idx_list.extend(preds)
                    
                except tf.errors.OutOfRangeError:
                    break

            target_list = [target.decode() for target in target_list]
            idx2word = pickle.load(open(idx2word_path, 'rb'))
            pred_list = [" ".join(trans_idx2sen(pred_idx_list[i], idx2word)).split("</s>", 1)[0].strip() + "\n" 
                for i in range(len(pred_idx_list))]
            bleu_score = cal_bleu(target_list, pred_list)
            Log.info("bleu score: {:3f}".format(bleu_score))

            return target_list, pred_list

        target_list, pred_list = dev_step()
        save_tgt_pred_sens(os.path.join(pred_path, 'test_tgt_pred.txt'), target_list, pred_list)


if __name__ == "__main__":
    FLAGS = get_args()
    Log.info("Parameters:")
    for attr, value in FLAGS.flag_values_dict().items():
        Log.info("{}: {}".format(attr, value))
    
    test_record_file = os.path.join(FLAGS.dataset_path, 'test.tfrecords')
    vocab_path = os.path.join(FLAGS.dataset_path, 'vocab.txt')
    pre_word2vec_path = os.path.join(FLAGS.dataset_path, 'w2v.pkl')
    idx2word_path = os.path.join(FLAGS.dataset_path, 'idx2word.pkl')
    evaluate(test_record_file, vocab_path, pre_word2vec_path, idx2word_path, FLAGS.res_path)
