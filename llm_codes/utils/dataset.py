import json
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from imblearn.under_sampling import RandomUnderSampler

class INDDataSetForMistral(Dataset):
    '''
        iteratively return the profile of each author 
    '''
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSetForMistral, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()
        train_keys = []
        labels = []
        
        num_0 = 0
        num_1 = 0
        for key in author_keys :
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                }) 
                labels.append(0)
                num_0 += 1
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)
                num_1 += 1
        # assert num_0 + num_1 == len(train_keys)
        # sampling_strategy = {
        #     0: num_0,  # 少数类保留全部
        #     1: 4 * num_0  # 多数类保留3倍于少数类
        # }
        # rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
        # keys_ids = list(range(0,len(train_keys)))
        # keys_ids = [ [x, 0] for x in keys_ids ]
        # sampled_keys, _ = rus.fit_resample(keys_ids, labels)
        # self.train_keys = [train_keys[i[0]] for i in sampled_keys]

        self.train_keys = train_keys
        random.shuffle(self.train_keys)
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,)

        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)

        # 4是因为有inst_begin_tokens， inst_end_tokens， bos_token_id， eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_len = self.max_source_length + self.max_target_length + 4


    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] +self.author[self.train_keys[index]['author']]['outliers']
        profile = [self.pub[p]['title'] for p in profile if p != self.train_keys[index]['pub']] #delete disambiguate paper
        random.shuffle(profile)
        # breakpoint()
        # limit context token lenth up to max_len - 500
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500: # left 500 for the instruction templete
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) #limit the disambiguate paper title token lenth
        context = self.instruct.format(profile_text,title)
        
        human_tokens = self.tokenizer.encode(text=context, add_special_tokens=False, truncation=True, max_length=self.max_source_length)
        assistant_tokens = self.yes_token if self.train_keys[index]['label'] else self.no_token

        input_tokens = self.inst_begin_tokens + human_tokens + self.inst_end_tokens
        output_tokens = assistant_tokens + [self.eos_token_id]

        input_ids = [self.bos_token_id] + input_tokens + output_tokens
        target_mask = [0] + [0] * len(input_tokens) + [1] * len(output_tokens)
        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[:self.max_seq_len]
        target_mask = target_mask[:self.max_seq_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        return {
            "input_ids": input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }

class INDDataSetForGLM(Dataset):
    '''
        iteratively return the profile of each author 
    '''
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSetForGLM, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()
        train_keys = []
        labels = []

        num_0 = 0
        num_1 = 0
        for key in author_keys :
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                }) 
                labels.append(0)
                num_0 += 1
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)
                num_1 += 1

        # assert num_0 + num_1 == len(train_keys)
        # sampling_strategy = {
        #     0: num_0,  # 少数类保留全部
        #     1: 4 * num_0  # 多数类保留3倍于少数类
        # }
        # rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
        rus = RandomUnderSampler(random_state=0)
        keys_ids = list(range(0,len(train_keys)))
        keys_ids = [ [x, 0] for x in keys_ids ]
        sampled_keys, _ = rus.fit_resample(keys_ids, labels)
        self.train_keys = [train_keys[i[0]] for i in sampled_keys]

        # self.train_keys = train_keys
        random.shuffle(self.train_keys)
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,)

        self.max_seq_len = self.max_source_length + self.max_target_length

    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] +self.author[self.train_keys[index]['author']]['outliers']
        profile = [self.pub[p]['title'] for p in profile if p != self.train_keys[index]['pub']] #delete disambiguate paper
        random.shuffle(profile)
        # breakpoint()
        # limit context token lenth up to max_len - 500
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500: # left 500 for the instruction templete
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) #limit the disambiguate paper title token lenth
        context = self.instruct.format(profile_text,title)
        
        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True, max_length=self.max_source_length)
        label_ids = self.yes_token if self.train_keys[index]['label'] else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids)-2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids":input_ids,
            "labels":labels,
            "author":self.train_keys[index]['author'],
            "pub":self.train_keys[index]['pub'],
        }

class IND4EVAL(Dataset):
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(IND4EVAL, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()

        self.val_set = []
        if 'normal_data' in self.author[list(author_keys)[0]]:
            for key in author_keys:   
                for pub_key in self.author[key]['normal_data']:   
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                        'label':1
                    }) 
                for pub_key in self.author[key]['outliers']:
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                        'label':0
                    }) 
        elif 'papers' in self.author[list(author_keys)[0]]:
            for key in author_keys:   
                for pub_key in self.author[key]['papers']:   
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                    }) 
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

    def __len__(self):
        return len(self.val_set)
    
    def __getitem__(self, index):
        if "normal_data" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['normal_data'] +self.author[self.val_set[index]['author']]['outliers']
        elif "papers" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['papers']
        else:
            raise("No profile found")
        profile = [self.pub[p]['title'] for p in profile if p != self.val_set[index]['pub']] #delete disambiguate paper
        random.shuffle(profile)

        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500:
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.val_set[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) 
        context = self.instruct.format(profile_text,title)
        return {
            "input_ids":context,
            "author":self.val_set[index]['author'],
            "pub":self.val_set[index]['pub'],
        }

class INDDataSetForGLM4(Dataset):
    '''
        iteratively return the profile of each author 
    '''
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSetForGLM4, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()
        train_keys = []
        labels = []
        
        num_0 = 0
        num_1 = 0
        for key in author_keys :
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                }) 
                labels.append(0)
                num_0 += 1
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)
                num_1 += 1
        rus = RandomUnderSampler(random_state=0)
        keys_ids = list(range(0,len(train_keys)))
        keys_ids = [ [x, 0] for x in keys_ids ]
        sampled_keys, _ = rus.fit_resample(keys_ids, labels)
        self.train_keys = [train_keys[i[0]] for i in sampled_keys]

        random.shuffle(self.train_keys)
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,)

        self.user_begin_tokens = tokenizer.encode('<|user|>\n', add_special_tokens=False)
        self.assistant_begin_tokens = tokenizer.encode('<|assistant|>\n', add_special_tokens=False)

        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_len = self.max_source_length + self.max_target_length + 5

    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] +self.author[self.train_keys[index]['author']]['outliers']
        profile = [self.pub[p]['title'] for p in profile if p != self.train_keys[index]['pub']] #delete disambiguate paper
        random.shuffle(profile)
        # breakpoint()
        # limit context token lenth up to max_len - 500
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500: # left 500 for the instruction templete
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) #limit the disambiguate paper title token lenth
        context = self.instruct.format(profile_text, title)
        
        human_tokens = self.tokenizer.encode(text=context, add_special_tokens=False, truncation=True, max_length=self.max_source_length)
        assistant_tokens = self.yes_token if self.train_keys[index]['label'] else self.no_token

        input_tokens = self.user_begin_tokens + human_tokens
        output_tokens = assistant_tokens + [self.eos_token_id]

        input_ids = [151331, 151333] + input_tokens + self.assistant_begin_tokens + output_tokens
        target_mask = [0, 0] + [0] * len(input_tokens + self.assistant_begin_tokens) + [1] * len(output_tokens)
        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[:self.max_seq_len]
        target_mask = target_mask[:self.max_seq_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        return {
            "input_ids": input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }