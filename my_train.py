# coding = utf-8

"""
模型训练
"""

import os

import numpy as np
import time
import pickle
import tensorflow as tf

import my_hyparams as hp
from my_data_helpers import get_record_parser, gen_batch_dataset, load_word2vec, get_vocab_size
from my_model import Model
from my_utils import get_args, trans_idx2sen, cal_bleu, save_tgt_pred_sens, Log


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


def train_model(train_record_file, valid_record_file, vocab_path, pre_word2vec_path, idx2word_path, res_path):
    """
    train model

    Args:
        train_record_file: str file path of train tf.recordfile
        valid_record_file: str file path of valid tf.recordfile
        pre_word_vec: tensor pretrained word embedding
        vocab_size: int size of vocabulary
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
    
    if FLAGS.use_pre_wordvec:
        pre_word2vec = load_word2vec(pre_word2vec_path)
    else:
        pre_word2vec = None

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
        model = Model(iterator, vocab_size, pre_word2vec, FLAGS)
        Log.info("build model success!")
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(lr_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
        train_op = optimizer.minimize(model.batch_avg_loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=3)
        
        # init model
        if FLAGS.reload_model:
            Log.info("reload model!")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        else:
            sess.run(tf.global_variables_initializer())
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("train/loss", model.batch_avg_loss)
        acc_summary = tf.summary.scalar("train/accuracy", model.batch_avg_acc)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(res_path, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
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

            _, step, summaries, batch_avg_loss, batch_avg_acc = sess.run(
                [train_op, global_step, train_summary_op, model.batch_avg_loss, model.batch_avg_acc], feed_dict)
            
            if step % 500 == 0:
                Log.info("Train Step: {:d} \t| loss: {:.3f} \t".format(step, batch_avg_loss))
            
            train_summary_writer.add_summary(summaries, step)


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
            target_list = []
            pred_idx_list = []
            acc = []

            while True:
                try:
                    feed_dict = {
                        handle: valid_handle,
                        model.dropout_rate: 0.0
                    }
                    step, batch_avg_loss, ppl, batch_avg_acc, target, preds = sess.run([global_step, model.batch_avg_loss, 
                        model.ppl, model.batch_avg_acc, model.target, model.preds], feed_dict)
                    
                    target_list.extend(target)
                    pred_idx_list.extend(preds)
                    loss_list.append(batch_avg_loss)
                    ppl_list.append(ppl)
                    acc.append(batch_avg_acc)
                    
                except tf.errors.OutOfRangeError:
                    break
            
            target_list = [target.decode() for target in target_list]
            idx2word = pickle.load(open(idx2word_path, 'rb'))
            pred_list = [" ".join(trans_idx2sen(pred_idx_list[i], idx2word)).split("</s>", 1)[0].strip() + "\n" 
                for i in range(len(pred_idx_list))]    
            mean_loss = np.mean(loss_list)
            mean_ppl = np.mean(ppl_list)
            mean_acc = np.mean(acc)
            bleu_score = cal_bleu(target_list, pred_list)
            Log.info("==================================")
            Log.info("Eval Step: {:d} \t| loss: {:.3f} \t| bleu: {:.3f}\t| ppl: {:3f}".format(
                step, mean_loss, bleu_score, mean_ppl))
            Log.info("==================================")
            
            summary_MeanLoss = tf.Summary(value=[tf.Summary.Value(tag="{}/mean_acc".format('dev'), 
                                        simple_value=np.mean(mean_acc))])
            summary_MeanAcc = tf.Summary(value=[tf.Summary.Value(tag='{}/mean_loss'.format('dev'), 
                                        simple_value=np.mean(mean_loss))])
            dev_summary_writer.add_summary(summary_MeanLoss, step)
            dev_summary_writer.add_summary(summary_MeanAcc, step)
      
            return mean_loss, bleu_score, target_list, pred_list
        
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
                mean_loss, mean_score, target_list, pred_list = dev_step()
            
                # early stop
                if mean_loss > optimal_loss:
                    early_break += 1
                else:
                    optimal_loss = mean_loss
                    early_break = 0
                if early_break > 4:
                    break

                # restore model
                if mean_score > optimal_score:
                    optimal_score = mean_score
                    save_tgt_pred_sens(os.path.join(pred_path, 'tgt_pred.txt'), target_list, pred_list)
                    Log.info("update best model start: model path = {}".format(model_path))
                    saver.save(sess, os.path.join(ckpt_path, 'model'), global_step=current_step)
                    Log.info("update model success!")
                Log.info("evaluate model success!")
        
        Log.info("trian model success!")
        

if __name__ == "__main__":
    FLAGS = get_args()

    Log.info("Parameters:")
    for attr, value in FLAGS.flag_values_dict().items():
        Log.info("{}: {}".format(attr, value))

    train_record_file = os.path.join(FLAGS.dataset_path, 'train.tfrecords')
    valid_record_file = os.path.join(FLAGS.dataset_path, 'valid.tfrecords')
    vocab_path = os.path.join(FLAGS.dataset_path, 'vocab.txt')
    pre_word2vec_path = os.path.join(FLAGS.dataset_path, 'w2v.pkl')
    idx2word_path = os.path.join(FLAGS.dataset_path, 'idx2word.pkl')
    res_path = os.path.join(FLAGS.res_path, str(int(time.time())))
    train_model(train_record_file, valid_record_file, vocab_path, pre_word2vec_path, idx2word_path, res_path)
