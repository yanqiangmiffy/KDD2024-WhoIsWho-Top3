#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/5/20 9:19
"""

import logging
import pickle

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def make_corpus(text_col='text'):
    train_author = pd.read_pickle('output/step1/step1_train_author.pkl')
    valid_author = pd.read_pickle('output/step1/step1_valid_author.pkl')
    corpus = train_author[text_col].values.tolist() + valid_author[text_col].values.tolist()
    with open(f'output/step2_corpus_{text_col}.txt', 'w', encoding='utf-8') as f:
        for line in corpus:
            if len(line.strip()):
                f.write(line + '\n')
    print(len(corpus))


def train_tfidf_model(
        corpus_file='output/step2_corpus.txt',
        output_dir='output/step2_tfidf_svd_pipeline_ngram12.joblib',
        n_components=64,
        n_iter=10,
        random_state=2024,
        save_model=True
):
    print("train_tfidf_model",corpus_file,output_dir)
    with open(corpus_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )),
        ('svd', TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state))
    ])

    # 训练并转换文本
    paper_features = pipe.fit_transform(texts)
    print(paper_features.shape)
    if save_model:
        joblib.dump(pipe, output_dir)

        # 获取并打印词表大小
        tfidf_vocab_size = len(pipe.named_steps['tfidf'].vocabulary_)
        print("Size of TfidfVectorizer vocabulary:", tfidf_vocab_size)


def train_count_model(
        corpus_file='output/step2_corpus.txt',
        output_dir='output/step2_tfidf_svd_pipeline_ngram12.joblib',
        n_components=64,
        n_iter=10,
        random_state=2024,
        save_model=True
):
    with open(corpus_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    pipe = Pipeline([
        ('count', CountVectorizer(
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
        )),
        ('svd', TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state))
    ])

    # 训练并转换文本
    paper_features = pipe.fit_transform(texts)
    print(paper_features.shape)
    if save_model:
        joblib.dump(pipe, output_dir)

        # 获取并打印词表大小
        tfidf_vocab_size = len(pipe.named_steps['count'].vocabulary_)
        print("Size of CountVectorizer vocabulary:", tfidf_vocab_size)


def train_word2vec(
        emb_size=64,
        model_path=None,
        mode='train',
        text_column='text'
):
    """
    生成Word2Vec特征
    :param df: 输入数据框
    :param value: 要训练的列名
    :param emb_size: 词向量维度大小
    :param model_path: 模型保存路径
    :param mode: 'train' 表示训练模式，'inference' 表示推理模式
    :return: 含有Word2Vec特征的数据框
    """
    df = pd.read_pickle('output/step1/step1_df.pkl')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("训练word2vec特征", text_column)
    # 将文本列转化为分词列表
    df[text_column] = df[text_column].apply(lambda x: x.split())
    sentences = df[text_column].tolist()
    print([sent for sent in sentences if len(sent) > 5][:2])
    if mode == 'train':
        model = Word2Vec(sentences, vector_size=emb_size, window=10, min_count=3, sg=0, hs=0, seed=2024, epochs=10,
                         workers=8)
        # 保存模型
        if model_path:
            model.save(model_path)
            print(f"模型已保存到 {model_path}")
            # 打印词表大小
        vocab_size = len(model.wv)
        print(f"词表大小: {vocab_size}")

if __name__ == '__main__':
    # make_corpus(text_col='text')
    # make_corpus(text_col='venue')
    # make_corpus(text_col='author_names_text')
    # make_corpus(text_col='org_names_text')


    for text_col in ['text','venue','author_names_text','org_names_text']:
        train_tfidf_model(
                corpus_file=f'output/step2/step2_corpus_{text_col}.txt',
                output_dir=f'output/step2/step2_tfidf12_svd64_{text_col}.joblib',
                n_components=64,
                n_iter=10,
                random_state=2024,
                save_model=True
        )

    for text_col in ['text','venue','author_names_text','org_names_text']:
        train_count_model(
                corpus_file=f'output/step2/step2_corpus_{text_col}.txt',
                output_dir=f'output/step2/step2_count12_svd64_{text_col}.joblib',
                n_components=64,
                n_iter=10,
                random_state=2024,
                save_model=True
        )

    for text_col in ['text','venue','author_names_text','org_names_text']:
        train_word2vec(
            emb_size=64,
            model_path=f'output/step2/step2_word2vec_{text_col}.bin',
            mode='train',
            text_column=text_col
        )