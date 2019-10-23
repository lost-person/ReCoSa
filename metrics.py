# coding = utf-8

import numpy as np
from nltk.translate import bleu_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import Log


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
    bleu_score_list = []
    for res, preds in zip(target_list, pred_list):
        score = bleu_score.sentence_bleu([res.split()], preds.split(), smoothing_function=bleu_score.SmoothingFunction().method1)
        bleu_score_list.append(score)
    
    return 100 * np.mean(bleu_score_list)


def cal_distinct(pred_list, n_gram=1):
    """
    calculate distinct
    """
    ngram_vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), decode_error="ignore", token_pattern = r'\b\w+\b')
    ngram_arr = ngram_vectorizer.fit_transform(pred_list).toarray()
    exist = (ngram_arr > 0) * 1.0
    factor = np.ones(ngram_arr.shape[1])
    dis_ngram_list = np.dot(exist, factor)
    sum_list = np.sum(ngram_arr, 1)
    distinct_list = dis_ngram_list / sum_list
    return np.mean(distinct_list)


def batch_trans_id2embed(batch_seq_id, word_embed):
    """
    transform batch sequence's idx to embedding

    Args:
        batch_seq_id: tensor batch sequence' word index, [batch_size, max_uttr_len]
        word_embed: tensor word embedding [vocab_size, embed_dim]
    """
    Log.info("transform batch sequence index to embedding start: batch_seq_id_shape = {}, word_embed_shape = {}".format(
        batch_seq_id.shape, word_embed.shape))
    batch_seq_embed_list = [trans_id2embed(seq_id, word_embed) for seq_id in batch_seq_id]
    Log.info("transform success!")
    return batch_seq_embed_list


def trans_id2embed(seq_id, word_embed):
    """
    transform one sequence's idx to embedding

    Args:
        seq_id: tensor sequence' word index, [max_uttr_len]
        word_embed: tensor word embedding [vocab_size, embed_dim]
    """
    Log.info("transform sequence index to embedding start: seq_id_shape = {}, word_embed_shape = {}".format(
        seq_id.shape, word_embed.shape))
    seq_embed_list = []
    for idx in seq_id:
        if idx == 3:
            break
        seq_embed_list.append(word_embed[idx])
    Log.info("transform success!")
    return seq_embed_list


def batch_embed_metrics(batch_res_seq_id, batch_pred_seq_id, word_embed):
    """
    batch embedding-based metrics

    Args:
        batch_res_seq_embed: tensor batch response sequence' idx,  [batch_size, max_uttr_len]
        batch_pred_seq_embed: tensor batch predict sequence' idx,  [batch_size, max_uttr_len]
        word_embed: tensor word embedding
    Return:
        score: float 
    """
    Log.info("embedding_based metrics start: batch_res_seq_shape = {}, batch_pred_seq_shape = {}".format(
        batch_res_seq_id.shape, batch_pred_seq_id.shape))
    
    greedy_score_list = []
    embed_avg_score_list = []
    vec_extrema_score_list = []
    # transform idx to word embedding
    batch_res_seq_embed = batch_trans_id2embed(batch_res_seq_id, word_embed)
    batch_pred_seq_embed = batch_trans_id2embed(batch_pred_seq_id, word_embed)

    embed_dim = word_embed.shape()[1]
    
    # evaluate start
    for res_seq_embed, pred_seq_embed in zip(batch_res_seq_embed, batch_pred_seq_embed):
        greedy_score_list.append(greedy_match(res_seq_embed, pred_seq_embed))
        embed_avg_score_list.append(embed_avg(res_seq_embed, pred_seq_embed))
        vec_extrema_score_list.append(vec_extrema(res_seq_embed, pred_seq_embed, embed_dim))
    
    greedy_mean_score = np.mean(greedy_score_list)
    embed_avg_mean_score = np.mean(embed_avg_score_list)
    vec_extrema_mean_score = np.mean(vec_extrema_score_list)
    Log.info("embedding_based metrics success: greedy_mean_score = {:3f}, embed_avg_mean_score = {:3f}, " + 
            "vec_extrema_mean_score = {:3f}".format(greedy_mean_score, embed_avg_mean_score, vec_extrema_mean_score))
    return greedy_mean_score, embed_avg_mean_score, vec_extrema_mean_score


def cal_greedy(embed_list1, embed_list2):
    """
    calculate the greedy score

    Args:
        embed_list1: list list of word embedding
        embed_list2: list list of word embedding
    """
    Log.info("calculate greedy score start: embed_list1_size = {}, embed_list2_size = {}".format(
        len(embed_list1), len(embed_list2)))
    score_list = []
    for embed1 in embed_list1:
        score_list.append(np.max([cosine_similarity(embed1, embed2) for embed2 in embed_list2]))
    socre = np.sum(score_list)
    socre = np.divide(socre, len(embed_list1))
    Log.info("calculate greedy success: score = {:3f}".format(socre))
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
    Log.info("calculate greedy matching start: res_embed_list_size = {}, pred_embed_list_size = {}".format(
        len(res_embed_list), len(pred_embed_list)))
    greedy1 = cal_greedy(res_embed_list, pred_embed_list)
    greedy2 = cal_greedy(pred_embed_list, res_embed_list)
    greedy = (greedy1 + greedy2) / 2
    Log.info("calculate greedy matching success: greedy = {}".format(greedy))
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
    Log.info("calculate embedding average start: res_embed_list_size = {}, pred_embed_list_size = {}".format(
        len(res_embed_list), len(pred_embed_list)))
    res_avg_embed = np.divide(np.sum(res_embed_list), np.linalg.norm(np.sum(res_embed_list)))
    pred_avg_embed = np.divide(np.sum(pred_embed_list), np.linalg.norm(np.sum(pred_embed_list)))
    score = cosine_similarity(res_avg_embed, pred_avg_embed)
    Log.info("calculate embedding average success: score = {:3f}".format(score))
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
    Log.info("calculate the maximum and minimum dimention value start: embed_list_size = {}, embed_dim = {}".format(
        len(embed_list, embed_dim)))
    
    extrema_embed = np.zeros([embed_dim])
    # convert to tensor
    embed_arr = np.array(embed_list)
    max_arr = np.max(embed_arr, axis=0)
    min_arr = np.min(embed_arr, axis=0)
    for i, (min_value, max_value) in enumerate(zip(min_arr, max_arr)):
        if max_value > abs(min_value):
            embed_arr[i] = max_value
        else:
            embed_arr[i] = min_value
    Log.info("calculate the maximum and minimum dimention value success!")
    return extrema_embed


def vec_extrema(res_embed_list, pred_embed_list):
    """
    calculate the vector extrema
    
    Args:
        res_embed_list: list list of response sequence' word embedding
        pred_embed_list: list list of predict sequence' word embedding
    Returns:
        score: float score of vector extrema
    """
    Log.info("calculate vector extrema start: res_embed_list_size = {}, pred_embed_list_size = {}".format(
        len(res_embed_list), len(pred_embed_list)))
    res_extrema_embed = cal_extrema_embed(res_embed_list)
    pred_extrema_embed = cal_extrema_embed(pred_embed_list)
    score = cosine_similarity(res_extrema_embed, pred_extrema_embed)
    Log.info("calculate vector extrema success: score = {:3f}".format(score))
    return score
