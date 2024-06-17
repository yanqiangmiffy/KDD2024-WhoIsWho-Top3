#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/3 0:58
"""
import warnings

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import gc
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings('ignore', category=RuntimeWarning)



def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(len(df.columns))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
def author_exists(author_name, authors):
    normalized_author_name = normalize_name(author_name)
    for author in authors:
        normalized_name = normalize_name(author['name'])
        if normalized_author_name & normalized_name:
            return 1
    return 0


def common_substring_length(s1, s2):
    # 初始化二维数组
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0

    # 填充dp数组
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                longest += 1

    return longest


def common_character_length(s1, s2):
    # 创建字符计数器
    from collections import Counter
    count1 = Counter(s1)
    count2 = Counter(s2)
    common_chars = count1 & count2
    return sum(common_chars.values())


def total_common_lengths(author_name, authors):
    """
    common_substring_length 函数计算两个字符串的公共子串长度。
    common_character_length 函数计算两个字符串的公共字符长度之和。
    total_common_lengths 函数遍历 authors 列表，计算并返回所有名字与 author_name 的公共子串长度之和和公共字符长度之和，
    以及author_name与authors中每个name公有长度的最大值。
    :param author_name:
    :param authors:
    :return:
    """
    normalized_author_name = author_name.lower().replace(" ", "")
    total_substring_length = 0
    total_character_length = 0
    max_common_length = 0

    for author in authors:
        normalized_name = author['name'].lower().replace(" ", "")
        substring_length = common_substring_length(normalized_author_name, normalized_name)
        character_length = common_character_length(normalized_author_name, normalized_name)

        total_substring_length += substring_length
        total_character_length += character_length
        max_common_length = max(max_common_length, substring_length, character_length)

    return total_substring_length, total_character_length, max_common_length


def has_token_intersection(author_name, authors):
    # 将author_name中的单词提取出来，并转换为小写
    tokens = set(author_name.lower().split())

    # 遍历每个author，检查名字中的单词是否与author_name中的单词存在交集
    for author in authors:
        name_tokens = set(author['name'].lower().split())
        if tokens & name_tokens:  # 检查交集
            return 1

    return 0


def extract_author_count(authors):
    try:
        authors_list = eval(authors)
        return len(authors_list)
    except:
        return 0


def extract_keywords_count(keywords):
    if pd.isna(keywords):
        return 0
    else:
        return len(keywords.split())


def extract_coauthors(authors, main_author):
    try:
        authors_list = authors
        coauthors = [author['name'] for author in authors_list if author['name'].lower() != main_author.lower()]
        return coauthors
    except:
        return []


from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, \
    minkowski, braycurtis, correlation, chebyshev, jensenshannon, mahalanobis, \
    seuclidean, sqeuclidean

def get_vector_sim(query_vec, title_vec, num):
    """
    Computes the similarity or distance between two vectors using a specified metric.

    Parameters:
    - query_vec (list or ndarray): The first vector.
    - title_vec (list or ndarray): The second vector.
    - num (int): Specifies the metric to use, where:
        1 = cosine,
        2 = canberra,
        3 = cityblock,
        ...
        12 = squared Euclidean.

    Returns:
    - float: The computed similarity or distance. Returns 0 on error.
    """
    distance_functions = {
        1: cosine,
        2: lambda u, v: canberra(u, v) / len(u),
        3: lambda u, v: cityblock(u, v) / len(u),
        4: euclidean,
        5: braycurtis,
        6: minkowski,
        7: correlation,
        8: chebyshev,
        9: jensenshannon,
        10: mahalanobis,
        11: seuclidean,
        12: sqeuclidean
    }

    try:
        # Fetch the appropriate function based on the user's choice
        func = distance_functions.get(num)
        if func:
            return func(query_vec, title_vec)
        else:
            raise ValueError("Invalid metric number provided.")
    except Exception as e:
        # Optionally, log or print the exception message if needed
        print(f"Error computing distance: {e}")
        return 0
