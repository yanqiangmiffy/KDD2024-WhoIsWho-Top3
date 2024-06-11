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
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from utils import *

tqdm.pandas()


def normalize_name(name):
    # 将名字转换为小写并去掉多余空格
    name = name.lower().strip()
    # 将名字拆分为姓和名，并处理颠倒的情况
    parts = name.split()
    if len(parts) == 2:
        return {f"{parts[0]} {parts[1]}", f"{parts[1]} {parts[0]}"}
    else:
        return {name}


def author_exists(author_name, authors):
    normalized_author_name = normalize_name(author_name)
    for author in authors:
        normalized_name = normalize_name(author['name'])
        if normalized_author_name & normalized_name:
            return 1
    return 0


# def process_text_data(group):
#     # Aggregate all texts into a single list per author
#     all_texts = group['text'].tolist()
#     combined_text = " ".join(all_texts)
#     words = combined_text.split()
#
#     # Count word frequencies and extract the top 100 words
#     word_count = Counter(words)
#     top100_words = set(word for word, _ in word_count.most_common(100))
#
#     # Function to calculate overlap scores
#     def calculate_overlap(text, compare_set):
#         text_words = set(text.split())
#         return len(text_words & compare_set)
#
#     # Apply the function to calculate overlaps
#     group['common_words_count'] = group['text'].apply(lambda x: calculate_overlap(x, set(words)))
#     group['top100_common_words_count'] = group['text'].apply(lambda x: calculate_overlap(x, top100_words))
#
#     return group

def process_text_data(group, column_name='text'):
    # Aggregate all texts into a single list per author from the specified column
    all_texts = group[column_name].tolist()
    combined_text = " ".join(all_texts)
    words = combined_text.split()

    # Count word frequencies and extract the top 100 words
    word_count = Counter(words)
    top100_words = set(word for word, _ in word_count.most_common(100))

    # Function to calculate overlap scores
    def calculate_overlap(text, compare_set):
        text_words = set(text.split())
        return len(text_words & compare_set)

    # Apply the function to calculate overlaps using the specified column
    group['common_words_count_' + column_name] = group[column_name].apply(lambda x: calculate_overlap(x, set(words)))
    group['top100_common_words_count_' + column_name] = group[column_name].apply(lambda x: calculate_overlap(x, top100_words))
    return group

import distance
def prepare_top_words(text, top_n=500):
    # 统计词频并返回最常见的top_n个词
    words = text.split()
    most_common_words = [word for word, freq in Counter(words).most_common(top_n)]
    return ' '.join(most_common_words)


def calculate_distances(data, text_columns, top_n=500):
    # 对每个作者的每个文本字段聚合并找出top 500词
    for column in text_columns:
        # 按author_id聚合文本，并提取top500次
        grouped_texts = data.groupby('author_id')[column].apply(lambda texts: ' '.join(texts.dropna()))
        top_words = grouped_texts.apply(lambda text: prepare_top_words(text, top_n))
        data = data.join(top_words.rename(f'author_{column}'), on='author_id')

    # 计算距离
    sim_func_dict = {"jaccard": distance.jaccard, "sorensen": distance.sorensen}
    for column in text_columns:
        for sim_func_name, sim_func in tqdm(sim_func_dict.items(), desc=f"距离特征-{column}"):
            # 使用try-except结构处理可能的错误
            data[f'{sim_func_name}_{column}'] = data.apply(
                lambda row: safe_distance_calc(row, sim_func_name, sim_func, column),
                axis=1
            )
            print(data[f'{sim_func_name}_{column}'].describe())

    return data
def safe_distance_calc(row, sim_func_name, sim_func, column):
    # 安全地计算距离，处理空集
    author_set = set(row[f'author_{column}'].split())
    text_set = set(row[column].split())
    if not author_set or not text_set:  # 检查任一集合是否为空
        return 0  # 如果为空，返回NaN或其他合理的默认值
    else:
        return sim_func(author_set, text_set)


def get_stats_features(data=None, mode='train'):
    print(data.shape)
    raw_cols = [col for col in data.columns]

    data['len_title'] = data['title'].apply(lambda x: len(x))
    # data['len_authors']=data['authors'].apply(lambda x:len(x))
    data['len_author_names'] = data['author_names'].apply(lambda x: len(x))
    data['len_org_names'] = data['org_names'].apply(lambda x: len(x))
    data['len_author_names_text'] = data['author_names_text'].apply(lambda x: len(x))
    data['len_org_names_text'] = data['org_names_text'].apply(lambda x: len(x))
    data['len_abstract'] = data['abstract'].apply(lambda x: len(x))
    data['len_keywords'] = data['abstract'].apply(lambda x: len(x))
    data['len_venue'] = data['venue'].apply(lambda x: len(x))
    data['len_text'] = data['text'].apply(lambda x: len(x))
    data['year_now'] = 2024 - data['year']

    data['ratio_title_to_text'] = data['len_title'] / data['len_text']
    data['ratio_abstract_to_text'] = data['len_abstract'] / data['len_text']
    data['ratio_keywords_to_text'] = data['len_keywords'] / data['len_text']
    data['ratio_author_names_to_org_names'] = data['len_author_names'] / data['len_org_names']
    data['ratio_author_names_text_to_org_names_text'] = data['len_author_names_text'] / data['len_org_names_text']
    data['diff_title_abstract'] = data['len_title'] - data['len_abstract']
    data['diff_authors_orgs'] = data['len_author_names'] - data['len_org_names']
    data['diff_text_abstract'] = data['len_text'] - data['len_abstract']
    data['product_title_abstract'] = data['len_title'] * data['len_abstract']
    data['product_authors_orgs'] = data['len_author_names'] * data['len_org_names']
    data['title_square'] = data['len_title'] ** 2
    data['abstract_root'] = np.sqrt(data['len_abstract'])
    data['ratio_abstract_to_text'] = data['len_abstract'] / (data['len_text'] + 0.001)

    data['is_author_exists'] = data.progress_apply(lambda row: author_exists(row['author_name'], row['authors']),
                                                   axis=1)
    data[['author_total_substring_length',
          'author_total_character_length',
          'author_max_common_length'
          ]] = data.progress_apply(
        lambda row: total_common_lengths(row['author_name'], row['authors']), axis=1, result_type='expand')
    data['author_has_token_intersection'] = data.progress_apply(
        lambda row: has_token_intersection(row['author_name'], row['authors']), axis=1)

    expand_cols=[
        'ratio_title_to_text','ratio_abstract_to_text',
        'ratio_title_to_text','ratio_abstract_to_text',
        'ratio_keywords_to_text','ratio_author_names_to_org_names',
        'ratio_author_names_text_to_org_names_text','diff_title_abstract',
        'diff_authors_orgs','diff_text_abstract',
        'product_title_abstract','product_authors_orgs',
        'title_square','abstract_root',
        'ratio_abstract_to_text','is_author_exists',
        'author_total_substring_length','author_total_character_length',
        'author_max_common_length','author_has_token_intersection',
    ]
    for col in expand_cols:
        group_col='author_id'
        data_group = data.groupby(group_col).agg({
            col: ['max', 'min', 'mean', 'std', 'sum'],
        })
        data_group.columns = pd.Index([f'{group_col}_' + e[0] + "_" + e[1] for e in data_group.columns.tolist()])
        data = data.merge(data_group, how='left', on=group_col)
        print(data.shape)
        del data_group

    base_cols = [
        ['author_id', 'year'],
        ['author_id', 'venue'],
    ]
    target_col = expand_cols
    for base_pair in base_cols:
        for col in tqdm(target_col):
            if col not in base_pair:
                data[f'group_{base_pair[0]}_{base_pair[1]}_{col}_mean'] = data[base_pair + [col]].groupby(base_pair)[
                    col].transform('mean')
                data[f'group_{base_pair[0]}_{base_pair[1]}_{col}_sum'] = data[base_pair + [col]].groupby(base_pair)[
                    col].transform('sum')

    # data = data.groupby('author_id').progress_apply(process_text_data)
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='text')
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='keywords')
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='title')
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='abstract')
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='author_names_text')
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='org_names_text')
    data = data.groupby('author_id').progress_apply(process_text_data, column_name='venue')

    text_columns = ['text', 'title', 'keywords', 'abstract', 'author_names_text', 'org_names_text']
    data = calculate_distances(data, text_columns)
    # 统计特征
    group_cols = [
        'author_id',
        'first_author',
        'second_author',
        'top1_author',
        'top2_author',
        'top1_keyword',
        'top2_keyword',
        'top1_org',
        'top2_org',
        'year',
        'venue'
    ]



    for group_col in group_cols:
        data_group = data.groupby(group_col).agg({
            'id': ['count', ],
            'venue': ['count', 'nunique'],
            'first_author': ['count', 'nunique'],
            'second_author': ['count', 'nunique'],
            'top1_keyword': ['count', 'nunique'],
            'top2_keyword': ['count', 'nunique'],
            'top1_author': ['count', 'nunique'],
            'top2_author': ['count', 'nunique'],
            'top1_org': ['count', 'nunique'],
            'top2_org': ['count', 'nunique'],
            'len_author_names': ['max', 'min', 'mean', 'std', 'sum'],
            'len_org_names': ['max', 'min', 'mean', 'std', 'sum'],
            'len_author_names_text': ['max', 'min', 'mean', 'std', 'sum'],
            'len_org_names_text': ['max', 'min', 'mean', 'std', 'sum'],
            'len_abstract': ['max', 'min', 'mean', 'std', 'sum'],
            'len_keywords': ['max', 'min', 'mean', 'std', 'sum'],
            'len_venue': ['max', 'min', 'mean', 'std', 'sum'],
            'len_text': ['max', 'min', 'mean', 'std', 'sum'],
            'year': ['max', 'min', 'mean', 'std', 'sum'],
        })
        data_group.columns = pd.Index([f'{group_col}_' + e[0] + "_" + e[1] for e in data_group.columns.tolist()])
        data = data.merge(data_group, how='left', on=group_col)
        print(data.shape)
        del data_group

    new_cols = ['author_id', 'id'] + [col for col in data.columns if col not in raw_cols]
    print("新增特征：", new_cols)
    data[new_cols].to_pickle(f'output/step3_{mode}_meta_features.pkl')
    print(data.shape)
    del data
    gc.collect()


if __name__ == '__main__':
    x = pd.read_pickle('output/step1/step1_df.pkl')
    train = x[x['label'] != -1].reset_index(drop=True)
    test = x[x['label'] == -1].reset_index(drop=True)
    # print(train.shape)
    # print(test.shape)

    get_stats_features(data=train.copy(), mode='train')
    get_stats_features(data=test.copy(), mode='test')
