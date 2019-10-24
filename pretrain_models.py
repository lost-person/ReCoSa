# coding = utf-8

"""
pretrained models
"""

import os
import pickle

from utils import Log


def load_word2vec(word_embed_path):
    """
    load word embedding(after look-up)

    Args:
        word_embed_path: str file path of word embedding
    Returns:
        pre_word_vec: Word2Vec pretrianed word embedding
    """
    if not os.path.exists(word_embed_path):
        Log.info("no data file exists: word2vec_path = {}".format(word_embed_path))
        return None
    
    Log.info("load word embedding start: word2vec_path = {}".format(word_embed_path))
    with open(word_embed_path, 'rb') as f:
        word_embed = pickle.load(f)
    Log.info("load word embedding success!")
    return word_embed


if __name__ == '__main__':
    pass
