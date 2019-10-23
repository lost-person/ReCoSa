# coding = utf-8

"""
preprocess dataset
"""

import os
import re
from collections import Counter
import hyparams as hp
import pickle
from utils import trans_sen2idx, Log


def gen_vocab(corpus_path, vocab_path, rebuild=False):
    """
    generate vocabulary with the corpus

    Args:
        corpus_path: str file path of corpus
        vocab_path: str file path of vocabulary
        rebuild: boolean rebuild vocabulary
    """
    if not os.path.exists(corpus_path):
        Log.info("no data file exists: corpus_path = {}".format(corpus_path))
        return
    
    if os.path.exists(vocab_path) and not rebuild:
        return

    Log.info("generate vocabulary start: corpus_path = {}, vocab_path = {}".format(corpus_path, vocab_path))
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = re.findall(r'\S+', f.read().lower())
    
    word2cnt = Counter(corpus)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write("{}\n{}\n{}\n{}\n".format('<pad>', '<unk>', '<s>', '</s>'))
        for word, _ in word2cnt.most_common(len(word2cnt)):
            f.write("{}\n".format(word))
        
    Log.info("generate the vocabulary success!")


def gen_word2idx_from_vocab(vocab_path, word2idx_path):
    """
    generate the word-idx dictionary

    Args:
        vacab_path: str path of word dictionary
        word2idx_path: str path of word2idx
    """
    if not os.path.exists(vocab_path):
        Log.info("no data file exists: data_path = {}".format(vocab_path))
        return None
    
    Log.info("gen word2idx start: vocab_path = {}, word2idx_path = {}".format(vocab_path, word2idx_path))
    word2idx = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word2idx[line.strip('\n')] = i
    
    with open(word2idx_path, 'wb') as f:
        pickle.dump(word2idx, f)

    Log.info("gen idx2word success!")


def gen_idx2word_from_vocab(vocab_path, idx2word_path):
    """
    generate the idx-word dictionary

    Args:
        vacab_path: str path of word dictionary
        idx2word_path: str path of idx2word
    Returns:
        idx2word: dict the inverse word dictionary
    """
    if not os.path.exists(vocab_path):
        Log.info("no data file exists: data_path = {}".format(vocab_path))
        return None
    
    Log.info("gen idx2word start: vocab_path = {}, idx2word_path = {}".format(vocab_path, idx2word_path))
    idx2word = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            idx2word[i] = line.strip('\n')
    
    with open(idx2word_path, 'wb') as f:
        pickle.dump(idx2word, f)
    
    Log.info("gen idx2word success!")


if __name__ == "__main__":
    vocab_path = os.path.join(hp.data_path, 'vocab.txt')
    if hp.data_path.find("ubuntu") == -1:
        gen_vocab(os.path.join(hp.data_path, 'train_data.txt'), vocab_path, True)
    gen_word2idx_from_vocab(os.path.join(hp.data_path, 'vocab.txt'), os.path.join(hp.data_path, 'word2idx.pkl'))
    gen_idx2word_from_vocab(os.path.join(hp.data_path, 'vocab.txt'), os.path.join(hp.data_path, 'idx2word.pkl'))
