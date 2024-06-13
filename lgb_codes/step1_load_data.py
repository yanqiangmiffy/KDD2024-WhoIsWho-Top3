#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/2 2:55
"""
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def seed_everything(seed):
    np.random.seed(seed)  # numpy的随机种子
    random.seed(seed)  # python内置的随机种子


seed_everything(42)
base_dir = '../data/IND-WhoIsWho/'
with open(base_dir + "train_author.json", 'r', encoding="utf-8") as f:
    train_author = json.load(f)
with open(base_dir + "pid_to_info_all.json", encoding="utf-8") as f:
    pid_to_info = json.load(f)
with open(base_dir + "ind_test_author_filter_public.json", encoding="utf-8") as f:
    valid_author = json.load(f)
with open(base_dir + "ind_test_author_submit.json", encoding="utf-8") as f:
    submission = json.load(f)

print("len(train_author),len(pid_to_info),len(valid_author),len(submission):\n")
print(len(train_author), len(pid_to_info), len(valid_author), len(submission))


def convert_paper_csv():
    """
    convet paper json to csv file
    :return:
    """
    papers = []
    for paper_id in pid_to_info:
        papers.append(pid_to_info[paper_id])
    df_paper = pd.DataFrame(papers)
    print(df_paper.shape)
    df_paper.to_csv(f"data/papers_info.csv", index=False)


def convert_author_csv(data, save_filename, is_train=True):
    author_samples = []
    for author in tqdm(data):
        if is_train:
            for paper_id in data[author]['normal_data']:
                author_samples.append(
                    {
                        'author_id': author,
                        'author_name': data[author]['name'],
                        'id': paper_id,
                        'label': 1,
                    }
                )
            for paper_id in data[author]['outliers']:
                author_samples.append(
                    {
                        'author_id': author,
                        'author_name': data[author]['name'],
                        'id': paper_id,
                        'label': 0,
                    }
                )
        else:
            for paper_id in data[author]['papers']:
                author_samples.append(
                    {
                        'author_id': author,
                        'author_name': data[author]['name'],
                        'id': paper_id,
                        'label': -1,
                    }
                )

    df_author = pd.DataFrame(author_samples)
    print(df_author['label'].value_counts())
    papers_info_df = pd.read_csv('data/papers_info.csv')
    df_author = df_author.merge(papers_info_df, how='left', on='id')
    df_author.to_csv(f"data/{save_filename}", index=False)
    df_author.head(5000).to_csv(f"data/sample_{save_filename}", index=False)


import string


def preprocess_text(text):
    """
    Process the input text by removing punctuation, converting to lowercase, and removing specified stopwords.

    :param text: A string containing the text to be processed.
    :return: A processed string with punctuation removed, converted to lowercase, and stopwords removed.
    """
    # Define stopwords
    stopwords = {'the', 'of', 'a', 'an', 'how', 'what', 'which', 'why', 'where', 'who', 'we', 'are', 'is', 'by', 'base',
                 'on', 'in'}

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert text to lowercase
    text = text.lower()

    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords)

    return text


def extract_first_two_authors_alternative(authors_list):
    """
    extract first_two_authors
    :param authors_list:
    :return:
    """
    # print(type(authors_str))
    try:
        # Safely evaluate string to literal, alternative to json.loads
        # authors_list = ast.literal_eval(authors_list)
        first_author = authors_list[0]['name'].lower() if len(authors_list) > 0 else ""
        second_author = authors_list[1]['name'].lower() if len(authors_list) > 1 else ""
    except Exception as e:
        print(f"Error processing: {authors_list} with error {e}")
        first_author, second_author = "", ""
    return first_author, second_author


def extract_names(data, key):
    """
    Function to extract author names and org names

    :param data:
    :param key:
    :return:
    """
    try:
        # Convert the string representation of list of dictionaries to an actual list of dictionaries
        # data = ast.literal_eval(data)
        # Extract the specific key values (name or org) from each dictionary in the list
        return [entry[key] for entry in data if entry[key]]
    except:
        # Return NaN if there's an issue with data conversion or extraction
        return pd.NA


def find_top_entities(group):
    """
    Function to identify top 2 authors and top 2 organizations, excluding self-references

    :param group:
    :return:
    """
    # Flatten the lists of names and organizations in the group
    all_authors = [author for sublist in group['author_names'] for author in sublist]
    all_orgs = [org for sublist in group['org_names'] for org in sublist]

    # Get the current author and their primary organization (if any)
    author_id = group['author_id'].iloc[0]
    own_name = group['author_name'].iloc[0].lower()

    # Count frequencies for authors and organizations
    author_counts = Counter(all_authors)
    org_counts = Counter(all_orgs)

    # Remove the author's own name and primary organization from the counts
    if own_name in author_counts:
        del author_counts[own_name]

    # Find the top two authors and organizations
    top_authors = author_counts.most_common(2)
    top_orgs = org_counts.most_common(2)

    # Prepare the outputs, handling cases where there might be fewer entries
    top1_author = top_authors[0][0] if top_authors else None
    top2_author = top_authors[1][0] if len(top_authors) > 1 else None
    top1_org = top_orgs[0][0] if top_orgs else None
    top2_org = top_orgs[1][0] if len(top_orgs) > 1 else None

    return pd.Series([top1_author, top2_author, top1_org, top2_org])


def fill_concat(df):
    """
    Preprocess and enhance a DataFrame with publication data by handling missing values,
    normalizing text, calculating top keywords for each author, and merging various textual attributes.

    Parameters:
    df (DataFrame): The input DataFrame containing columns like 'title', 'abstract', 'keywords', 'author_id', etc.

    Returns:
    DataFrame: The modified DataFrame with additional columns for processed text and top keywords.
    """
    df['title'] = df['title'].fillna('').apply(lambda x: str(x).lower())
    df['abstract'] = df['abstract'].fillna('').apply(lambda x: str(x).lower())
    df['keywords'] = df['keywords'].fillna('[]').apply(eval)
    df['keywords'] = df['keywords'].apply(lambda x: [keyword.lower() for keyword in x])
    df['keywords_list'] = df['keywords'].copy()
    # 聚合每个作者的所有关键词
    grouped_keywords = df.groupby('author_id')['keywords'].sum()
    # 计算每个作者的top1和top2关键词
    top_keywords = grouped_keywords.apply(lambda keywords: Counter(keywords).most_common(2))
    # 创建两个新列，分别存储top1和top2关键词
    df['top1_keyword'] = df['author_id'].map(top_keywords.apply(lambda x: x[0][0] if len(x) > 0 else None))
    df['top2_keyword'] = df['author_id'].map(top_keywords.apply(lambda x: x[1][0] if len(x) > 1 else None))
    # print(df[['author_id', 'top1_keyword', 'top2_keyword']].head())

    # 聚合每个作者的所有关键词
    grouped_keywords = df.groupby('author_id')['keywords'].sum()
    # 计算每个作者的top 100关键词
    top_keywords = grouped_keywords.apply(lambda keywords: Counter(keywords).most_common(400))

    # 创建一个新列，存储每个作者的top 100关键词列表
    df['top_words'] = df['author_id'].map(top_keywords.apply(lambda x: [keyword for keyword, count in x]))
    df['top_words'] = df['top_words'].apply(lambda x: ' '.join(x))
    # 查看结果
    # print(df[['author_id', 'top_words']])

    df['venue'] = df['venue'].fillna('VENUE_NAN').apply(lambda x: str(x).lower())
    df['year'] = df['year'].fillna(2024).astype(int)
    df['authors'] = df['authors'].apply(lambda x: eval(x))
    df['abstract'] = df['abstract'].fillna('').apply(lambda x: str(x).lower()).apply(preprocess_text)
    df['keywords'] = df['keywords'].apply(lambda x: ' '.join(x))
    df['keywords'] = df['keywords'].apply(lambda x: str(x).lower())
    df['text'] = df['title'] + ' ' + df['abstract'] + ' ' + df['keywords']
    df["text"] = df["text"].astype(str).apply(preprocess_text)
    df[['first_author', 'second_author']] = df['authors'].apply(
        lambda x: extract_first_two_authors_alternative(x)).apply(pd.Series)
    df['author_names'] = df['authors'].apply(lambda x: extract_names(x, 'name'))
    df['org_names'] = df['authors'].apply(lambda x: extract_names(x, 'org'))
    df['author_names'] = df['author_names'].apply(lambda names: [name.lower() for name in names])
    df['org_names'] = df['org_names'].apply(lambda orgs: [org.lower() for org in orgs])

    df['author_names_text'] = df['author_names'].apply(lambda x: ' '.join(x))
    df['org_names_text'] = df['org_names'].apply(lambda x: ' '.join(x))

    top_entities = df.groupby('author_id').apply(find_top_entities)
    top_entities.columns = ['top1_author', 'top2_author', 'top1_org', 'top2_org']
    df = df.merge(top_entities, on='author_id', how='left')
    # print(df['keywords_list'])
    return df


def process_author():
    """
       Main function to process training and validation author data. Reads data, processes it using 'fill_concat',
       and saves the processed data to disk. Additionally, combines training and validation data for further analysis.
    """
    train_author = pd.read_csv('data/train_author.csv')
    valid_author = pd.read_csv('data/valid_author.csv')
    print(train_author.shape)
    print(valid_author.shape)
    train_author = fill_concat(train_author.copy())
    valid_author = fill_concat(valid_author.copy())

    train_author.to_pickle('output/step1/step1_train_author.pkl')
    valid_author.to_pickle('output/step1/step1_valid_author.pkl')
    print(train_author.columns)

    df = pd.concat([train_author, valid_author], axis=0).reset_index(drop=True)
    df.to_pickle('output/step1/step1_df.pkl')
    df.head(2000).to_csv('output/step1/step1_df.csv', index=False)
    print(df.shape)


if __name__ == '__main__':
    convert_paper_csv()
    convert_author_csv(train_author, 'train_author.csv')
    convert_author_csv(valid_author, 'valid_author.csv', is_train=False)
    process_author()
