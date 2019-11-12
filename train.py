# coding = utf-8

"""
模型训练
"""

import os

import numpy as np
import time
import pickle
import tensorflow as tf

import hyparams as hp
from data_helpers import get_record_parser, gen_batch_dataset
from model import Model
from utils import get_args, save_tfsummary, Log
from vocab import get_vocab_size, load_idx2word, trans_idxs2sen
from metrics import cal_bleu, save_tgt_pred_sens, cal_distinct, embed_metrics
from pretrain_models import load_word2vec


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


def train_model(train_record_file, valid_record_file, vocab_path, idx2word_path, res_path):
    """
    train model

    Args:
        train_record_file: str file path of train tf.recordfile
        valid_record_file: str file path of valid tf.recordfile
        vocab_path: str file path of vocabulary
        idx2word_path: str file path of idx -> word
        res_path: str file path of model's results
    """
    # create path that restore results of model
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    ckpt_path = os.path.join(res_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    pred_path = os.path.join(res_path, 'pred')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    # get size of vocab and pretrained word embedding
    vocab_size = get_vocab_size(vocab_path)
    if not vocab_size:
        Log.info("get vocab_size err: vocab_path = {}".format(vocab_path))
        return

    # load data
    Log.info("load train and valid dataset start!")
    train_dataset = load_tfrecord(train_record_file, FLAGS, True)
    valid_dataset = load_tfrecord(valid_record_file, FLAGS, False)
    if not train_dataset or not valid_dataset:
        Log.info("load train and valid dataset err!")
        return
    Log.info("load train and valid dataset success!")

    session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # iterate dataset
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        valid_iterator = valid_dataset.make_initializable_iterator()
        train_handle = sess.run(train_iterator.string_handle())
        
        Log.info("build model start!")
        model = Model(iterator, vocab_size, FLAGS)
        Log.info("build model success!")
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(lr_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
        train_op = optimizer.minimize(model.loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=3)
        
        # init model
        if FLAGS.reload_model:
            Log.info("reload model!")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        else:
            sess.run(tf.global_variables_initializer())
        
        # Train Summaries
        train_summary_dir = os.path.join(res_path, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
        train_summary_writer.add_graph(sess.graph)
        
        # Dev summaries
        dev_summary_dir = os.path.join(res_path, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)


        def train_step():
            """
            a single training step
            """        
            train_step = tf.train.global_step(sess, global_step)  
            feed_dict = {
                lr_rate: FLAGS.lr_rate,
                handle: train_handle, 
                model.dropout_rate: FLAGS.dropout_rate
            }

            _, step, loss, acc = sess.run([train_op, global_step, model.loss, model.acc], feed_dict)
            save_tfsummary(train_summary_writer, step, 'train/acc', acc)
            save_tfsummary(train_summary_writer, step, 'train/loss', loss)

            if step % FLAGS.print_step == 0:
                Log.info("Train Step: {:d} \t| loss: {:.3f}".format(step, loss))


        def dev_step():
            """
            a single dev step

            Returns:
                mean_los: float mean loss on valid dataset
                bleu_score: float mean bleu score on valid dataset
            """
            sess.run(valid_iterator.initializer)
            valid_handle = sess.run(valid_iterator.string_handle())
            
            loss_list = []
            ppl_list = []
            tgt_list = []
            res_idx_list = []
            pred_idx_list = []
            acc_list = []

            while True:
                try:
                    feed_dict = {
                        handle: valid_handle,
                        model.dropout_rate: 0.0
                    }
                    step, loss, ppl, acc, target, res, preds = sess.run([global_step, model.loss, 
                        model.ppl, model.acc, model.target, model.response, model.preds], feed_dict)

                    tgt_list.extend(target)
                    res_idx_list.extend(res)
                    pred_idx_list.extend(preds)
                    loss_list.append(loss)
                    ppl_list.append(ppl)
                    acc_list.append(acc)
                    
                except tf.errors.OutOfRangeError:
                    break
            
            tgt_list = [target.decode() for target in tgt_list]
            idx2word = load_idx2word(idx2word_path)
            pred_list = [trans_idxs2sen(pred_idx_list[i], idx2word).split("</s>", 1)[0].strip()
                for i in range(len(pred_idx_list))]    
            loss = np.mean(loss_list)
            ppl = np.mean(ppl_list)
            mean_acc = np.mean(acc)
            bleu_score = cal_bleu(tgt_list, pred_list)
            dist1 = cal_distinct(pred_list)
            dist2 = cal_distinct(pred_list, 2)
            Log.info("=" * 40)
            Log.info("loss: {:.3f} | bleu: {:.3f} | ppl: {:.3f} | dist_1 = {:.3f}, dist_2 = {:.3f}".format(
                loss, bleu_score, ppl, dist1, dist2))
            Log.info("=" * 40)
            
            save_tfsummary(dev_summary_writer, step, 'dev/acc', acc)
            save_tfsummary(dev_summary_writer, step, 'dev/loss', loss)
            save_tfsummary(dev_summary_writer, step, 'dev/ppl', ppl)
            save_tfsummary(dev_summary_writer, step, 'dev/bleu', bleu_score)
            save_tfsummary(dev_summary_writer, step, 'dev/dist1', dist1)
            save_tfsummary(dev_summary_writer, step, 'dev/dist2', dist2)
      
            return loss, bleu_score, tgt_list, pred_list
        
        early_break = 0
        optimal_loss = 100000
        optimal_score = 0
        model_path = os.path.join(ckpt_path, 'model')

        Log.info("train model start!")
        for step in range(FLAGS.num_epochs):
            train_step()
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.eval_step == 0:
                Log.info("evaluate model start!")
                cur_loss, cur_score, target_list, pred_list = dev_step()

                # restore model
                if cur_score > optimal_score:
                    optimal_score = cur_score
                    save_tgt_pred_sens(os.path.join(pred_path, 'tgt_pred.txt'), target_list, pred_list)
                    Log.info("update best model start: model path = {}".format(model_path))
                    saver.save(sess, os.path.join(ckpt_path, 'model'), global_step=current_step)
                    Log.info("update model success!")
                Log.info("evaluate model success!")

                # early stop
                if cur_loss > optimal_loss:
                    early_break += 1
                else:
                    optimal_loss = cur_loss
                    early_break = 0
                if early_break > 4:
                    break
        
        Log.info("trian model success!")
        

if __name__ == "__main__":
    FLAGS = get_args()

    Log.info("Parameters:")
    for attr, value in FLAGS.flag_values_dict().items():
        Log.info("{}: {}".format(attr, value))

    train_record_file = os.path.join(FLAGS.data_path, 'train.tfrecords')
    valid_record_file = os.path.join(FLAGS.data_path, 'valid.tfrecords')
    vocab_path = os.path.join(FLAGS.data_path, 'vocab.txt')
    idx2word_path = os.path.join(FLAGS.data_path, 'idx2word.pkl')
    res_path = os.path.join(FLAGS.res_path, str(int(time.time())))
    train_model(train_record_file, valid_record_file, vocab_path, idx2word_path, res_path)
