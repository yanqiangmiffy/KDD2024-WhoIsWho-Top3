#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/2 11:55
"""

import gc
from collections import Counter

import joblib
import pandas as pd
from gensim.models import Word2Vec
from scipy.stats import skew, kurtosis
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils import *

tqdm.pandas()


def generate_features_for_paper(
        data,
        model_path='',
        emb_size=64,
        mode='train',
        text_col='text',
        feature_type='tfidf',
        group_by_col='author_id',
        only_sim=False,
):
    """
    General function to generate various types of text features for papers, including NLP embeddings.

    Parameters:
        data (pd.DataFrame): DataFrame containing the text data.
        emb_size (int): Embedding size (not used for sentence_transformers).
        mode (str): Operation mode ('train' or 'test').
        text_col (str): Text column for feature generation.
        feature_type (str): Type of feature ('tfidf', 'count', 'word2vec', 'sentence_transformers').
        group_by_col (str): Column name for grouping data.
    """
    print(model_path, emb_size, mode, text_col, feature_type, group_by_col)
    print(f"Generating {feature_type} features for paper, grouped by {group_by_col}")
    print(data.shape)

    if feature_type in ['tfidf', 'count']:
        pipe = joblib.load(model_path)
        features = pipe.transform(data[text_col])
    elif feature_type == 'word2vec':
        model = Word2Vec.load(model_path)
        print(f"Model loaded from {model_path}")
        emb_matrix = []
        for seq in tqdm(data[text_col], total=len(data)):
            vec = [model.wv[w] for w in seq.split() if w in model.wv]
            emb_matrix.append(np.mean(vec, axis=0) if vec else [0] * emb_size)
        features = np.array(emb_matrix)
    elif feature_type in ['sbert','deberta','scibert']:
        print(model_path)
        model = SentenceTransformer(model_path)
        print("Loading SentenceTransformer model...")
        features = model.encode(data[text_col].tolist(), show_progress_bar=True,batch_size=512)
        # 使用 PCA 将特征降维到 64 维
        pca = PCA(n_components=64)
        features = pca.fit_transform(features)
        print("Reduced features shape:", features.shape)
        feature_type=feature_type+'_'+model_path.split('/')[-1]
        print(feature_type)
    else:
        raise ValueError("Unsupported feature type. Choose from 'tfidf', 'count', 'word2vec', 'sentence_transformers'.")
    # Create a DataFrame from the extracted features
    feature_prefix = f"{feature_type}_{group_by_col}_{text_col}"
    if feature_type == 'sentence_transformers':
        feature_columns = [f"{feature_prefix}_{i}" for i in range(features.shape[1])]
    else:
        feature_columns = [f"{feature_prefix}_{i}" for i in range(emb_size)]
    paper_features = pd.DataFrame(features, columns=feature_columns)
    if group_by_col == 'author_id':
        result_paper_df = pd.concat([data[[group_by_col, 'id']], paper_features], axis=1)
    else:
        result_paper_df = pd.concat([data[['author_id', 'id', group_by_col]], paper_features], axis=1)
    # mean_vectors = result_paper_df.groupby(group_by_col)[feature_columns].transform('mean')
    # result_paper_df[feature_columns] = mean_vectors

    mean_vectors = result_paper_df.groupby(group_by_col)[feature_columns].transform('mean').values
    paper_vectors = result_paper_df[feature_columns].values
    # 定义相似度类型
    similarity_types = {
        'cosine': 1,
        'canberra': 2, 'cityblock': 3, 'euclidean': 4,
        'braycurtis': 5, 'minkowski': 6, 'correlation': 7,
        'chebyshev': 8, 'jensenshannon': 9, 'sqeuclidean': 12
    }
    # 为每种距离类型计算相似度
    for sim_name, sim_num in similarity_types.items():
        # 初始化一个用于存储相似度结果的列
        result_paper_df[f'vec_{sim_name}_{feature_prefix}_similarity'] = 0
        # 遍历每一行来计算相似度
        for i in tqdm(range(len(result_paper_df)), desc=sim_name):
            # 提取单个论文向量和对应的均值向量
            query_vec = mean_vectors[i]
            title_vec = paper_vectors[i]
            # 使用之前定义的相似度计算函数
            similarity = get_vector_sim(query_vec, title_vec, sim_num)
            # 存储计算结果
            result_paper_df.at[i, f'vec_{sim_name}_{feature_prefix}_similarity'] = similarity

    # 输出结果以查看
    del mean_vectors, paper_vectors

    if not only_sim:
        for stat in [
            'mean','max','min'
        ]:
            stat_vectors = result_paper_df.groupby(group_by_col)[feature_columns].transform(stat)
            stats_feature_columns = [f"{feature_prefix}_{stat}_{i}" for i in range(emb_size)]
            result_paper_df[stats_feature_columns] = stat_vectors
            diff_feature_columns = [f"{feature_prefix}_{stat}_{i}_diff" for i in range(emb_size)]
            result_paper_df[diff_feature_columns] = result_paper_df[feature_columns] - stat_vectors
            del stat_vectors
            result_paper_df.drop(columns=stats_feature_columns, inplace=True)
            gc.collect()
    result_paper_df.drop(columns=feature_columns, inplace=True)
    print(result_paper_df.columns[:20])
    print(result_paper_df.shape)
    # Save the result to a pickle file
    output_path = f'output/step3/step3_{mode}_paper_{feature_prefix}_features.pkl'
    result_paper_df = reduce_mem(result_paper_df)
    result_paper_df.to_pickle(output_path)
    print(f"Features saved to {output_path}")
    print(result_paper_df.head())
    # Cleanup
    del paper_features, features, result_paper_df, data
    gc.collect()


def get_text_embedding(data=None, mode='train'):
    data['author_year']=data['author_id'].astype(str)+'_'+data['year'].astype(str)
    data['author_venue']=data['author_id'].astype(str)+'_'+data['venue'].astype(str)
    # 论文内容文本 emb
    for text_col in [
        'text',
        'title',
        'keywords',
        'abstract'
    ]:
        for feature_type in [
            'scibert',
            'deberta',
            'tfidf',
            'count',
            'word2vec',
        ]:
            for group_col in [
                'author_id',
                # 'first_author',
                # 'second_author',
                # 'top1_author',
                # 'top2_author',
                'top1_keyword',
                'top1_org',
                # 'top2_org'
                'author_year',
                'author_venue'
            ]:
                if feature_type == 'tfidf':
                    model_path = 'output/step2/step2_tfidf12_svd64_text.joblib'
                elif feature_type == 'count':
                    model_path = 'output/step2/step2_count12_svd64_text.joblib'
                elif feature_type == 'word2vec':
                    model_path = 'output/step2/step2_word2vec_text.bin'
                elif feature_type == 'deberta':
                    model_path = 'I:/pretrained_models/bert/english/deberta-v3-base'
                elif feature_type == 'scibert':
                    model_path = 'I:/pretrained_models/bert/english/scibert_scivocab_uncased'
                else:
                    model_path=''
                generate_features_for_paper(
                    data.copy(),
                    model_path=model_path,
                    emb_size=64,
                    mode=mode,
                    text_col=text_col,
                    feature_type=feature_type,
                    group_by_col=group_col,
                    only_sim=True
                )

    # 作者embedding
    for text_col in ['venue']:
        for feature_type in [
            'tfidf',
            'count', 'word2vec'
        ]:
            for group_col in [
                'author_id',
                # 'first_author',
                # 'second_author',
                # 'top1_author',
                # 'top2_author',
                'top1_keyword',
                'top1_org',
                # 'top2_org'
            ]:
                if feature_type == 'tfidf':
                    model_path = 'output/step2/step2_tfidf12_svd64_venue.joblib'
                elif feature_type == 'count':
                    model_path = 'output/step2/step2_count12_svd64_venue.joblib'
                elif feature_type == 'word2vec':
                    model_path = 'output/step2/step2_word2vec_venue.bin'
                else:
                    model_path = ''
                generate_features_for_paper(
                    data.copy(),
                    model_path=model_path,
                    emb_size=64,
                    mode=mode,
                    text_col=text_col,
                    feature_type=feature_type,
                    group_by_col=group_col
                )
    for text_col in ['author_names_text']:
        for feature_type in [
            'tfidf',
            'count', 'word2vec'
        ]:
            for group_col in [
                'author_id',
                # 'first_author',
                # 'second_author',
                # 'top1_author',
                # 'top2_author',
                'top1_keyword',
                'top1_org',
                # 'top2_org'
            ]:
                if feature_type == 'tfidf':
                    model_path = 'output/step2/step2_tfidf12_svd64_author_names_text.joblib'
                elif feature_type == 'count':
                    model_path = 'output/step2/step2_count12_svd64_author_names_text.joblib'
                elif feature_type == 'word2vec':
                    model_path = 'output/step2/step2_word2vec_author_names_text.bin'
                else:
                    model_path = ''
                generate_features_for_paper(
                    data.copy(),
                    model_path=model_path,
                    emb_size=64,
                    mode=mode,
                    text_col=text_col,
                    feature_type=feature_type,
                    group_by_col=group_col
                )
    for text_col in ['org_names_text']:
        for feature_type in [
            'tfidf',
                             'count',
                             'word2vec'
        ]:
            for group_col in [
                'author_id',
                # 'first_author',
                # 'second_author',
                # 'top1_author',
                # 'top2_author',
                'top1_keyword',
                'top1_org',
                # 'top2_org'
            ]:
                if feature_type == 'tfidf':
                    model_path = 'output/step2/step2_tfidf12_svd64_org_names_text.joblib'
                elif feature_type == 'count':
                    model_path = 'output/step2/step2_count12_svd64_org_names_text.joblib'
                elif feature_type == 'word2vec':
                    model_path = 'output/step2/step2_word2vec_org_names_text.bin'
                else:
                    model_path = ''
                generate_features_for_paper(
                    data.copy(),
                    model_path=model_path,
                    emb_size=64,
                    mode=mode,
                    text_col=text_col,
                    feature_type=feature_type,
                    group_by_col=group_col
                )

if __name__ == '__main__':
    x = pd.read_pickle('output/step1/step1_df.pkl')
    train = x[x['label'] != -1].reset_index(drop=True)
    test = x[x['label'] == -1].reset_index(drop=True)
    # get_text_embedding(data=train.copy(), mode='train')
    get_text_embedding(data=test.copy(), mode='test')
