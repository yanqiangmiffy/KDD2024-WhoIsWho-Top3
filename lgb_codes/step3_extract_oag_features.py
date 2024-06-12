#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/8 12:44
"""
# https://github.com/THUDM/CogDL/tree/master

import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch
from cogdl.oag import oagbert
from utils import get_vector_sim, reduce_mem
import torch
from torch.nn.utils.rnn import pad_sequence
df = pd.read_csv('output/step1/step1_df.csv')
print(df.head())
print(df.columns)

# load time
tokenizer, model = oagbert("./oagbert-v2-sim")
model.eval()
model.to('cuda')
print(model)


def extract_oag_features(df):
    features = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        # Paper 1
        title = row['title']
        abstract = row['abstract']
        author_names = row['author_names']
        venue = row['venue']
        org_names = row['org_names']
        keywords = row['keywords_list']
        if idx<1:
            print(title, abstract, author_names, venue, org_names, keywords)
        # encode first paper
        input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(
            title=title, abstract=abstract, venue=venue, authors=author_names, concepts=keywords, affiliations=org_names
        )
        _, paper_embed_1 = model.bert.forward(
            input_ids=torch.LongTensor(input_ids).unsqueeze(0).to('cuda'),
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to('cuda'),
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to('cuda'),
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=torch.LongTensor(position_ids).unsqueeze(0).to('cuda'),
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to('cuda')
        )
        features.append(paper_embed_1.detach().cpu().numpy())
        del paper_embed_1, input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans
        del title, abstract, author_names, venue, org_names, keywords
    features = np.vstack(features)
    return features



def extract_oag_features_batch(df, batch_size=32):
    all_features = []

    # Process the dataframe in batches
    for start_idx in tqdm(range(0, len(df), batch_size),total=len(df)//batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        # Lists to collect inputs for padding
        batch_input_ids = []
        batch_input_masks = []
        batch_token_type_ids = []
        batch_position_ids = []
        batch_position_ids_second = []

        # Build inputs for the entire batch
        for _, row in batch_df.iterrows():
            inputs = model.build_inputs(
                title=row['title'],
                abstract=row['abstract'],
                venue=row['venue'],
                authors=row['author_names'],
                concepts=row['keywords'],
                affiliations=row['org_names']
            )
            # Collect inputs
            batch_input_ids.append(torch.LongTensor(inputs[0]))
            batch_input_masks.append(torch.LongTensor(inputs[1]))
            batch_token_type_ids.append(torch.LongTensor(inputs[2]))
            batch_position_ids.append(torch.LongTensor(inputs[4]))
            batch_position_ids_second.append(torch.LongTensor(inputs[5]))

        # Pad all collected inputs
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True).to('cuda')
        batch_input_masks = pad_sequence(batch_input_masks, batch_first=True).to('cuda')
        batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True).to('cuda')
        batch_position_ids = pad_sequence(batch_position_ids, batch_first=True).to('cuda')
        batch_position_ids_second = pad_sequence(batch_position_ids_second, batch_first=True).to('cuda')

        # Forward pass for the batch
        with torch.no_grad():
            _, batch_embeddings = model.bert(
                input_ids=batch_input_ids,
                token_type_ids=batch_token_type_ids,
                attention_mask=batch_input_masks,
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=batch_position_ids,
                position_ids_second=batch_position_ids_second
            )

        # Append batch embeddings to all_features
        all_features.append(batch_embeddings.cpu().numpy())
        # Clear memory to avoid leakage
        del batch_embeddings, batch_input_ids, batch_input_masks, batch_token_type_ids, batch_position_ids, batch_position_ids_second

    # Concatenate all features into a single numpy array
    features = np.vstack(all_features)
    return features

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
    # features = extract_oag_features(data)
    features = extract_oag_features_batch(data,batch_size=256)
    np.save(f'{mode}_fea.npy',features)
    # print(features)
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
            'mean', 'max', 'min'
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
        # 'title',
        # 'keywords',
        # 'abstract'
    ]:
        for feature_type in [
            # 'scibert',
            # 'deberta',
            # 'tfidf',
            # 'count',
            # 'word2vec',
            'oag'
        ]:
            for group_col in [
                'author_id',
                # 'author_year',
                # 'author_venue'
            ]:
                generate_features_for_paper(
                    data.copy(),
                    model_path="./oagbert-v2-sim",
                    emb_size=768,
                    mode=mode,
                    text_col=text_col,
                    feature_type=feature_type,
                    group_by_col=group_col,
                    only_sim=True
                )



if __name__ == '__main__':
    x = pd.read_pickle('output/step1/step1_df.pkl')
    train = x[x['label'] != -1].reset_index(drop=True)
    test = x[x['label'] == -1].reset_index(drop=True)
    get_text_embedding(data=train.copy(), mode='train')
    get_text_embedding(data=test.copy(), mode='test')
