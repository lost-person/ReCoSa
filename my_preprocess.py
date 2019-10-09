# coding = utf-8

"""
preprocess dataset
"""

import os
import re
from collections import Counter
import hyperparameters as hp
import pickle
import utils

log = utils.get_log(hp.log_conf)


def gen_vocab(corpus_path, vocab_path, rebuild=False):
    """
    generate vocabulary with the corpus

    Args:
        corpus_path: str file path of corpus
        vocab_path: str file path of vocabulary
        rebuild: boolean rebuild vocabulary
    """
    if not os.path.exists(corpus_path):
        log.info("no data file exists: corpus_path = {}".format(corpus_path))
        return
    
    if os.path.exists(vocab_path) and not rebuild:
        return

    log.info("generate vocabulary start: corpus_path = {}, vocab_path = {}".format(corpus_path, vocab_path))
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = re.findall(r'\S+', f.read().lower())
    
    word2cnt = Counter(corpus)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write("{}\n{}\n{}\n{}\n".format('<pad>', '<unk>', '<s>', '</s>'))
        for word, _ in word2cnt.most_common(len(word2cnt)):
            f.write("{}\n".format(word))
        
    log.info("generate the vocabulary success!")


def gen_word2idx_from_vocab(vocab_path, word2idx_path):
    """
    generate the word-idx dictionary

    Args:
        vacab_path: str path of word dictionary
        word2idx_path: str path of word2idx
    """
    if not os.path.exists(vocab_path):
        log.info("no data file exists: data_path = {}".format(vocab_path))
        return None
    
    log.info("gen word2idx start: vocab_path = {}, word2idx_path = {}".format(vocab_path, word2idx_path))
    word2idx = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word2idx[line.split("\t")[0]] = i
    
    with open(word2idx_path, 'wb') as f:
        pickle.dump(word2idx, f)

    log.info("gen idx2word success!")


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
        log.info("no data file exists: data_path = {}".format(vocab_path))
        return None
    
    log.info("gen idx2word start: vocab_path = {}, idx2word_path = {}".format(vocab_path, idx2word_path))
    idx2word = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            idx2word[i] = line.split("\t")[0]
    
    with open(idx2word_path, 'wb') as f:
        pickle.dump(idx2word, f)
    
    log.info("gen idx2word success!")


def prepro_ubuntu_data(org_path, tgt_path, tgt_context_path, tgt_res_path):
    """
    preprocess the ubuntu dataset into standard format

    org format: data format: label\tcontext\treposonse
    standard format: data format: context\treposnse

    Args:
        org_path: str path of original dataset
        tgt_path: str path of dataset which has been preprocessed
    """
    if not os.path.exists(org_path):
        log.info("no data file exists: data_path = {}".format(org_path))
        return None

    data_list = []
    context_list = []
    res_list = []

    log.info("load data file start: data_path = {}".format(org_path))
    with open(org_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            eg_list = line.strip().split('\t')
            if len(eg_list) < 3:
                log.info("invalid data: line = {}".format(line))
                continue

            label = eg_list[0]
            if label == str(0):
                continue

            context = eg_list[1:-1]          
            response = eg_list[-1]

            context_list.append('\t'.join(context) + '\n')
            res_list.append(response + '\n')
            data_list.append('\t'.join(context) + '\t' + response + '\n')
    
    log.info("load data file success!")
    
    if not os.path.exists(os.path.dirname(tgt_path)):
        os.makedirs(tgt_path)

    log.info("save data start: file_path = {}, size = {}".format(tgt_path, len(data_list)))
    with open(tgt_path, 'w', encoding='utf-8') as f:
        f.writelines(data_list)
    
    with open(tgt_context_path, 'w', encoding='utf-8') as f:
        f.writelines(context_list)

    with open(tgt_res_path, 'w', encoding='utf-8') as f:
        f.writelines(res_list)
    log.info("save data success!")

def test():
    with open(os.path.join(hp.ubuntu_data_path, 'vocab.txt'), 'r', encoding='utf-8') as f:
        vocab = f.readlines()
    vocab = ["<pad>\t100000\n", "<unk>\t100000\n", "<s>\t100000\n", "</s>\t100000\n"] + vocab
    with open(os.path.join(hp.ubuntu_data_path, 'vocab_tmp.txt'), 'w', encoding='utf-8') as f:
        f.writelines(vocab)

if __name__ == "__main__":
    prepro_ubuntu_data(os.path.join(hp.ubuntu_data_path, 'train.txt'), os.path.join(
        hp.ubuntu_data_path, 'train_data.txt'), os.path.join(hp.ubuntu_res_path, 'out', 'train_context.txt'),
        os.path.join(hp.ubuntu_res_path, 'out', 'train_res.txt'))
    prepro_ubuntu_data(os.path.join(hp.ubuntu_data_path, 'valid.txt'), os.path.join(
        hp.ubuntu_data_path, 'valid_data.txt'), os.path.join(hp.ubuntu_res_path, 'out', 'valid_context.txt'),
        os.path.join(hp.ubuntu_res_path, 'out', 'valid_res.txt'))
    prepro_ubuntu_data(os.path.join(hp.ubuntu_data_path, 'test.txt'), os.path.join(
        hp.ubuntu_data_path, 'test_data.txt'), os.path.join(hp.ubuntu_res_path, 'out', 'test_context.txt'),
        os.path.join(hp.ubuntu_res_path, 'out', 'test_res.txt'))
    corpus_path = os.path.join(hp.ubuntu_data_path, 'train_data.txt')
    # gen_vocab(corpus_path, os.path.join(hp.ubuntu_data_path, 'vocab_tmp.txt'), True)
    gen_word2idx_from_vocab(os.path.join(hp.ubuntu_data_path, 'vocab.txt'), os.path.join(hp.ubuntu_data_path, 
                            'word2idx.pkl'))
    gen_idx2word_from_vocab(os.path.join(hp.ubuntu_data_path, 'vocab.txt'), os.path.join(hp.ubuntu_data_path, 
                            'idx2word.pkl'))
