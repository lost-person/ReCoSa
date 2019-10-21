# coding = utf-8

"""
模型
"""

import tensorflow as tf

from modules import *
from my_utils import Log

class Model():
    """
    A class that construct model

    """
    def __init__(self, batch, vocab_size, pre_word2vec, FLAGS):
        """
        initial the model

        Args:
            batch: tf.data
            vocab_size: int size of vocabulary
            FLAGS: tf.flags arguments
            pre_word2vec: tensor pretrained word embedding
        """
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        
        if FLAGS.is_training:
            self.context, self.context_len, self.response, self.source, self.target = batch.get_next()
        else:
            self.context = tf.placeholder(tf.int32, shape=(None, FLAGS.max_turn, FLAGS.max_uttr_len))
            self.context_len = tf.placeholder(tf.int32, shape=(None, FLAGS.max_turn))
            self.response = tf.placeholder(tf.int32, shape=(None, FLAGS.max_uttr_len))
        
        # 2 -> <s>
        self.decoder_inputs = tf.concat((tf.ones_like(self.response[:, : 1]) * 2, self.response[:, : -1]), -1)
        # self.word_embed = get_token_embeddings(vocab_size, FLAGS.embed_dim, pre_word2vec, FLAGS.use_pre_wordvec)

        # encoder
        with tf.variable_scope("encoder"):
            # [batch_size * max_turn, max_uttr_len, emb_dim]
            self.enc_embed = embedding(tf.reshape(self.context, [-1, FLAGS.max_uttr_len]), 
                                      vocab_size = vocab_size, 
                                      num_units = FLAGS.embed_dim, 
                                      scale = True,
                                      scope = "enc_embed")

            self.seq_len = tf.reshape(self.context_len, [-1])

            rnn_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_dim)
            self.uttn_outputs, self.uttn_states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.enc_embed, 
                sequence_length=self.seq_len, dtype=tf.float32, swap_memory=True)
            self.enc = tf.reshape(self.uttn_states, [-1, FLAGS.max_turn, FLAGS.hidden_dim])

            self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.context)[1]), 0), [tf.shape(self.context)[0], 1]),
                                      vocab_size=FLAGS.max_uttr_len, 
                                      num_units=FLAGS.hidden_dim, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
            self.enc = tf.layers.dropout(self.enc, rate=self.dropout_rate, 
                        training=tf.convert_to_tensor(FLAGS.is_training))
            
            # blocks
            for i in range(FLAGS.num_blocks):
                with tf.variable_scope("blocks_{}".format(i)):
                    # context self-attention
                    self.enc, _ = multihead_attention(queries=self.enc, keys=self.enc, num_units=FLAGS.hidden_dim,
                                                    num_heads=FLAGS.num_heads, dropout_rate=self.dropout_rate, 
                                                    is_training=FLAGS.is_training, causality=False)
                    self.enc = feedforward(self.enc, [4 * FLAGS.hidden_dim, FLAGS.hidden_dim])

        # decoder
        with tf.variable_scope("decoder"):
            # [batch_size, max_uttr_len, embed_dim]
            
            self.dec = embedding(self.decoder_inputs, vocab_size=vocab_size, num_units=FLAGS.hidden_dim,
                                scale=True,  scope="dec_embed")
            # postion embedding and dropout
            # self.dec += positional_encoding(self.decoder_inputs, vocab_size=FLAGS.max_uttr_len, 
            #                                 num_units=FLAGS.hidden_dim, zero_pad=False, 
            #                                 scale=False, scope="dec_pe")
            self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=FLAGS.max_uttr_len, 
                                      num_units=FLAGS.hidden_dim, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
            self.dec = tf.layers.dropout(self.dec, rate=self.dropout_rate, 
                training=tf.convert_to_tensor(FLAGS.is_training))

            # blocks
            for i in range(FLAGS.num_blocks):              
                with tf.variable_scope("blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # response self-attention
                    self.dec, _ = multihead_attention(queries=self.dec, keys=self.dec, 
                                                    num_units=FLAGS.hidden_dim, num_heads=FLAGS.num_heads,
                                                    dropout_rate=self.dropout_rate, is_training=FLAGS.is_training,
                                                    causality=True, scope="self_attention"
                    )
                
                    # vanilla attention
                    self.dec, self.attn = multihead_attention(queries=self.dec, keys=self.enc, 
                                                    num_units=FLAGS.hidden_dim, num_heads=FLAGS.num_heads,
                                                    dropout_rate=self.dropout_rate, is_training=FLAGS.is_training, 
                                                    causality=False,scope="vanilla_attention")
                    self.dec = feedforward(self.dec, [4 * FLAGS.hidden_dim, FLAGS.hidden_dim])
        
        # pred
        self.logits = tf.layers.dense(self.dec, vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.response, 0))
        self.batch_avg_acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.response)) * self.istarget) / (tf.reduce_sum(self.istarget))

        # loss
        if FLAGS.is_training:
            self.res_smoothed = label_smoothing(tf.one_hot(self.response, depth=vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.res_smoothed, logits=self.logits)
            self.batch_avg_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            self.ppl = tf.exp(self.batch_avg_loss)

if __name__ == "__main__":
    pass