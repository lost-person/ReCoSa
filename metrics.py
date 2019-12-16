# coding = utf-8

import sys
import numpy as np
from nltk.translate import bleu_score
from sklearn.feature_extraction.text import CountVectorizer

from utils import Log

def save_infer(infer_res_path, src_list, tgt_list, pred_list):
    """
    save results of infer

    Args:
        infer_res_path: str path of infer results
        src_list: list list of source
        tgt_list: list list of target
        pred_idx_list: list of pred
    """
    Log.info("save infer results start: tgt_pred_path = {}".format(infer_res_path))
    with open(infer_res_path, 'w', encoding='utf-8') as f:
        for src, target, pred in zip(src_list, tgt_list, pred_list):
            f.write("- src: " + src + "\n")
            f.write("- tgt: " + target + "\n")
            f.write("- pred: " + pred + "\n")
            f.write('\n')
    Log.info("save infer results success!")


def save_tgt_pred_sens(tgt_pred_path, target_list, pred_list):
    """
    存储预测的句子

    Args:
        tgt_pred_path: str 真实和预测结果的存储路径
        target_list: list 真实结果列表
        pred_idx_list: list 预测结果列表
    """
    Log.info("save tgt and pred start: tgt_pred_path = {}".format(tgt_pred_path))
    with open(tgt_pred_path, 'w', encoding='utf-8') as f:
        for target, pred in zip(target_list, pred_list):
            f.write("- tgt: " + target + "\n")
            f.write("- pred: " + pred + "\n")
    Log.info("save tgt and pred success!")


def cal_bleu(target_list, pred_list):
    """"
    计算bleu得分

    Args:
        target_list: list 真实结果列表
        pred_idx_list: list 预测结果列表
    Returns:
        bleu_score: float bleu得分
    """
    Log.info("calculate bleu score start: data_size = {}".format(len(target_list)))
    bleu_score_list = []
    for res, preds in zip(target_list, pred_list):
        score = bleu_score.sentence_bleu([res.split()], preds.split(), smoothing_function=bleu_score.SmoothingFunction().method1)
        bleu_score_list.append(score)
    Log.info("calculate bleu score success")
    return 100 * np.mean(bleu_score_list)


def cal_distinct(pred_list, n_gram=1):
    """
    calculate distinct

    Args:
        pred_id_list: list
        n-gram: int n-gram default 1
    Returns:
        distinct_score: float dist-n score
    """
    Log.info("calculate distinct score start: pred_id_list_size = {}, n_gram = {}".format(len(pred_list), n_gram))
    ngram_vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), decode_error="ignore", token_pattern = r'\b\w+\b')
    ngram_arr = ngram_vectorizer.fit_transform(pred_list).toarray()
    exist = (ngram_arr > 0) * 1.0
    factor = np.ones(ngram_arr.shape[1])
    dis_ngram_arr = np.dot(exist, factor)
    sum_arr = np.sum(ngram_arr, 1)
    indics = sum_arr != 0
    sum_arr = sum_arr[indics]
    dis_ngram_arr = dis_ngram_arr[indics]
    # sum_arr[sum_arr == 0] = sys.maxsize
    distinct_arr = dis_ngram_arr / sum_arr
    distinct_score = np.mean(distinct_arr)
    Log.info("calculate distinct score success")
    return distinct_score


def trans_list_id2embed(seq_id_list, word_embed):
    """
    transform batch sequence's idx to embedding

    Args:
        seq_id_list: tensor batch sequence' word index, [batch_size, max_uttr_len]
        word_embed: tensor word embedding [vocab_size, embed_dim]
    """
    Log.info("transform sequence idx to embedding start: data_size = {}".format(len(seq_id_list)))
    batch_seq_embed_list = [trans_id2embed(word_id_list, word_embed) for word_id_list in seq_id_list]
    Log.info("transform success!")
    return batch_seq_embed_list


def trans_id2embed(word_id_list, word_embed):
    """
    transform one sequence's idx to embedding

    Args:
        word_id_list: tensor sequence' word index, [max_uttr_len]
        word_embed: tensor word embedding [vocab_size, embed_dim]
    """
    seq_embed_list = []
    for word_id in word_id_list:
        if word_id == 3:
            break
        seq_embed_list.append(word_embed[word_id])
    return seq_embed_list


def embed_metrics(res_seq_id_list, pred_seq_id_list, word_embed):
    """
    batch embedding-based metrics

    Args:
        res_seq_id_list: list batch response sequence' idx,  [batch_size, max_uttr_len]
        pred_seq_id_list: list batch predict sequence' idx,  [batch_size, max_uttr_len]
        word_embed: tensor word embedding
    Return:
        score: float 
    """
    Log.info("embedding_based metrics start: data_size = {}".format(len(res_seq_id_list)))
    
    greedy_score_list = []
    embed_avg_score_list = []
    vec_extrema_score_list = []
    # transform idx to word embedding
    res_seq_embed_list = trans_list_id2embed(res_seq_id_list, word_embed)
    pred_seq_embed_list = trans_list_id2embed(pred_seq_id_list, word_embed)

    embed_dim = word_embed.shape[1]
    
    # evaluate start
    for res_seq_embed, pred_seq_embed in zip(res_seq_embed_list, pred_seq_embed_list):
        greedy_score_list.append(greedy_match(res_seq_embed, pred_seq_embed))
        embed_avg_score_list.append(embed_avg(res_seq_embed, pred_seq_embed))
        vec_extrema_score_list.append(vec_extrema(res_seq_embed, pred_seq_embed, embed_dim))
    
    greedy_mean_score = np.mean(greedy_score_list)
    embed_avg_mean_score = np.mean(embed_avg_score_list)
    vec_extrema_mean_score = np.mean(vec_extrema_score_list)
    Log.info("embedding_based metrics success")
    return greedy_mean_score, embed_avg_mean_score, vec_extrema_mean_score


def cal_greedy(embed_list1, embed_list2):
    """
    calculate the greedy score

    Args:
        embed_list1: list list of word embedding
        embed_list2: list list of word embedding
    """
    score_list = []
    for embed1 in embed_list1:
        score_list.append(np.max([cal_cosine_similarity(embed1, embed2) for embed2 in embed_list2]))
    socre = np.mean(score_list)
    return socre


def greedy_match(res_embed_list, pred_embed_list):
    """
    calculate the greedy matching

    Args:
        res_embed_list: list response sequence' word embedding
        pred_embed_list: list predict sequence' word embedding
    Returns:
        score: float score of greedy match
    """
    greedy1 = cal_greedy(res_embed_list, pred_embed_list)
    greedy2 = cal_greedy(pred_embed_list, res_embed_list)
    greedy = (greedy1 + greedy2) / 2
    return greedy


def embed_avg(res_embed_list, pred_embed_list):
    """
    calculate the embeddign average
    
    Args:
        res_embed_list: list list of response sequence' word embedding
        pred_embed_list: list list of predict sequence' word embedding
    Returns:
        score: float score of embedding average
    """
    res_avg_embed = np.divide(np.sum(res_embed_list, 0), np.linalg.norm(np.sum(res_embed_list, 0)))
    pred_avg_embed = np.divide(np.sum(pred_embed_list, 0), np.linalg.norm(np.sum(pred_embed_list, 0)))
    score = cal_cosine_similarity(res_avg_embed, pred_avg_embed)
    return score


def cal_extrema_embed(embed_list, embed_dim):
    """
    For each dimension of the word vectors, take the most extreme value amongst all word vectors in the sentence, and 
    use that value in the sentence-level embedding

    Args:
        embed_list: list list of sequence' word embedding
        embed_dim: int dimention of word embedding
    Returns:
        extrema_embed: array extrema embedding
    """
    # convert to tensor
    embed_arr = np.array(embed_list)
    max_arr = np.max(embed_arr, axis=0)
    min_arr = np.min(embed_arr, axis=0)
    extrema_embed = np.where(np.greater_equal(max_arr, abs(min_arr)), max_arr, min_arr)
    return extrema_embed


def vec_extrema(res_embed_list, pred_embed_list, embed_dim):
    """
    calculate the vector extrema
    
    Args:
        res_embed_list: list list of response sequence' word embedding
        pred_embed_list: list list of predict sequence' word embedding
        embed_dim: int word embedding
    Returns:
        score: float score of vector extrema
    """
    res_extrema_embed = cal_extrema_embed(res_embed_list, embed_dim)
    pred_extrema_embed = cal_extrema_embed(pred_embed_list, embed_dim)
    score = cal_cosine_similarity(res_extrema_embed, pred_extrema_embed)
    return score


def cal_cosine_similarity(vec_a, vec_b):
    """
    calcluate cosine similarity between vec_a and vec_b

    Args:
        vec_a: array vector a
        vec_b: array vector b
    Returns
        cosine_similarity: float cosine similarity
    """
    cosine_similarity = np.divide(np.dot(vec_a, vec_b), (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
    cosine_similarity = cosine_similarity * 0.5 + 0.5
    return cosine_similarity

if __name__ == "__main__":
    cal_distinct(["i i i", "i i"])
    cal_distinct(["i i i am zhanglu", "i love kv"], 2)
