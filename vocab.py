import os
import re
from collections import Counter
import hyparams as hp
import pickle
from utils import Log


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


def get_vocab_size(vocab_path):
    """
    get the size of vocabulary

    Args:
        vocab_path: str path of vocabulary
    Return:
        vocab_size: int size of vocabulary
    """
    if not os.path.exists(vocab_path):
        Log.info("no data file exists: vocab_path = {}".format(vocab_path))
        return None

    Log.info("get size of vocabulary start: vocab_path = {}".format(vocab_path))
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_size = len(f.readlines())
    
    Log.info("get size of vocabulary success: size = {}".format(vocab_size))
    return vocab_size


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
            word2idx[line.strip("\n")] = i
    
    with open(word2idx_path, 'wb') as f:
        pickle.dump(word2idx, f)

    Log.info("gen word2idx success: word2idx_size = {}".format(len(word2idx)))


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
            idx2word[i] = line.strip("\n")
    
    with open(idx2word_path, 'wb') as f:
        pickle.dump(idx2word, f)
    
    Log.info("gen idx2word success: idx2word_size = {}".format(len(idx2word)))

def trans_sen2idxs(seq, word2idx):
    """
    transforms sentence to idx list

    Args:
        seq: str sequence
        word2idx: dict word dictionary
    Returns:
        idx_list: list list of index
    """
    idx_list = [word2idx.get(word, 1) for word in seq.split(' ')]
    return idx_list


def trans_idxs2sen(idx_list, idx2word):
    """
    transforms idx_list to sentence

    Args:
        idx_list: list list of idx
        idx2word: dict idx2word dictionary
    Returns:
        seq: str sequence
    """
    seq = ' '.join([idx2word.get(idx, '<unk>') for idx in idx_list if idx not in [
            0, 2, 3]])
    return seq


if __name__ == "__main__":
    vocab_path = os.path.join(hp.data_path, 'vocab.txt')
    gen_word2idx_from_vocab(vocab_path, os.path.join(hp.data_path, 'word2idx.pkl'))
    gen_idx2word_from_vocab(vocab_path, os.path.join(hp.data_path, 'idx2word.pkl'))
