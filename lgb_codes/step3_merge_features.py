#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/2 23:52
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import gc
import os.path
import warnings
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)

tqdm.pandas()

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

def gen_train_test(data=None, mode='train'):
    data['author_year'] = data['author_id'].astype(str) + '_' + data['year'].astype(str)
    data['author_venue'] = data['author_id'].astype(str) + '_' + data['venue'].astype(str)
    for text_col in [
        'text',
        'title',
        'keywords',
        'abstract'
        'venue',
        'author_names_text',
        'org_names_text',
    ]:
        for feature_type in [
            'deberta',
            'tfidf',
            'count',
            'word2vec',
            'sbert',
            'scibert',
            'oag'
        ]:
            for group_col in [
                # 'author_year',
                # 'author_venue',
                'author_id',
                'top1_keyword',
                'top1_org',
                # 'first_author',
                # 'second_author',
                # 'top1_author',
                # 'top2_author',

                # 'top2_org',

            ]:
                if feature_type=='sbert':
                    feature_type=f'sbert_deberta-v3-base'
                    feature_prefix = f"{feature_type}_{group_col}_{text_col}"
                elif feature_type == 'scibert':
                    feature_type = f'scibert_scibert_scivocab_uncased'
                    feature_prefix = f"{feature_type}_{group_col}_{text_col}"
                else:
                    feature_prefix = f"{feature_type}_{group_col}_{text_col}"

                fea_path=f'output/step3/step3_{mode}_paper_{feature_prefix}_features.pkl'
                train_fea_path=f'output/step3/step3_train_paper_{feature_prefix}_features.pkl'
                test_fea_path=f'output/step3/step3_test_paper_{feature_prefix}_features.pkl'



                if os.path.exists(train_fea_path) and os.path.exists(test_fea_path):
                    if feature_type=='oag':
                        print("存在")
                        print(fea_path)
                        print(train_fea_path)
                        print(test_fea_path)

                    fea_path=(fea_path)
                    fea_df = pd.read_pickle(fea_path)
                    fea_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    if group_col!='author_id':
                        data = data.merge(fea_df, how='left', on=['author_id', 'id',group_col])
                    else:
                        data = data.merge(fea_df, how='left', on=['author_id', 'id'])
                    del fea_df
                    gc.collect()
                    print("data.shape===>",data.shape)

    # data=reduce_mem(data)
    fea_path=f'output/step3_{mode}_meta_features.pkl'
    fea_df = pd.read_pickle(fea_path)
    print("无穷大值检查:")
    print(fea_df.isin([np.inf, -np.inf]).sum())
    # 替换无穷大值为NaN
    fea_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fea_df=reduce_mem(fea_df)
    data = data.merge(fea_df, how='left', on=['author_id', 'id'])
    del fea_df

    cols=['first_author', 'second_author','top1_author', 'top2_author', 'top1_org', 'top2_org','venue']
    for col in cols:
        lb = LabelEncoder()
        data[col] = lb.fit_transform(data[col])

    del data['author_year']
    del data['author_venue']
    data.to_pickle(f'output/{mode}.pkl')
    data.head(100).to_csv(f'output/{mode}.csv', index=False)


if __name__ == '__main__':
    x = pd.read_pickle('output/step1/step1_df.pkl')
    train = x[x['label'] != -1].reset_index(drop=True)
    test = x[x['label'] == -1].reset_index(drop=True)
    # print(train.shape)
    # print(test.shape)
    gen_train_test(train.copy(), 'train')
    gen_train_test(test.copy(), 'test')
