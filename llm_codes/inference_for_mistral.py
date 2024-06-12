# -*- coding: utf-8 -*-

import os
from peft import PeftModel,get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch.utils.data import Dataset
import json
import random
from accelerate import Accelerator
from tqdm import tqdm
from utils.metric import compute_metric
import argparse
import torch.distributed as dist
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from trl import DPOTrainer, get_kbit_device_map

class IND4EVAL(Dataset):
    def __init__(self, dataset, tokenizer, max_source_length):
        super(IND4EVAL, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
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

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,)

        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)

        # 4是因为有inst_begin_tokens， inst_end_tokens， bos_token_id， eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

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

        human_tokens = self.tokenizer.encode(text=context, add_special_tokens=False, truncation=False)
        input_ids = [self.bos_token_id] + self.inst_begin_tokens + human_tokens + self.inst_end_tokens
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(attention_mask)
        return {
            "input_ids":input_ids,
            "attention_mask": attention_mask,
            "author":self.val_set[index]['author'],
            "pub":self.val_set[index]['pub'],
        }

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', help='The path to the lora file',default=None)
parser.add_argument('--model_path',default='ZhipuAI/chatglm3-6b-32k')
parser.add_argument('--pub_path', help='The path to the pub file',default=None)
parser.add_argument('--eval_path',default='eval_data.json')
parser.add_argument('--saved_dir',default='eval_result')
parser.add_argument('--batch_size',default='1')
parser.add_argument('--max_source_length',default='4096')
#bvac
parser.add_argument("--d_train_dir",type=str,default=None,)
parser.add_argument("--d_pre_model_dir",type=str,default=None,)
parser.add_argument("--d_model_dir",type=str,default=None,)
parser.add_argument("--d_summary_dir",type=str,default=None,)
parser.add_argument("--d_result_dir",type=str,default=None,)
args = parser.parse_args()

if not dist.is_initialized():
    print(f'dist.init_process_group=nccl')
    dist.init_process_group(backend='nccl')


if args.lora_path is not None:
    checkpoint = args.lora_path.split('/')[-1]
else:
    checkpoint = args.model_path.split('/')[-1]

accelerator = Accelerator()
device = torch.device(0)

batch_size = int(args.batch_size)
max_source_length = int(args.max_source_length)

model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

if args.lora_path is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, padding_side="left", trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
print(f'tokenizer special_tokens_map: {tokenizer.special_tokens_map}')

if args.lora_path is not None:
    model = PeftModel.from_pretrained(model, args.lora_path)
    print('done loading peft model')
YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with open(args.pub_path, "r" , encoding = "utf-8") as f:
    pub_data = json.load(f)
with open(args.eval_path, "r", encoding="utf-8") as f: 
    eval_data = json.load(f)

eval_dataset = IND4EVAL(
    (eval_data,pub_data),
    tokenizer,
    max_source_length = max_source_length,
)
print('done reading dataset')

def collate_fn(batch):
    lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
    # 取出batch中的最大长度，如果超过max_source_length，则取max_source_length
    # batch_max_len = min(max(lengths), max_source_length)
    batch_max_len = max(lengths)

    input_ids_batch, attention_mask_batch = [], []
    for x in batch:
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        padding_len = batch_max_len - len(input_ids)

        #推理时需要left padding
        attention_mask = [0] * padding_len + attention_mask
        input_ids = [tokenizer.pad_token_id] * padding_len + input_ids

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)

    input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
    attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)

    batch = {k: [item[k] for item in batch] for k in ('author', 'pub')}
    batch_input = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch
        }
    return batch_input, batch['author'], batch['pub']

dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size ,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)
model.eval()
result = []


YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with torch.no_grad():
    for index,batch in tqdm(enumerate(val_data)):
        batch_input, author, pub = batch

        if not hasattr(model, 'module'):
            response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 16 + 4, return_dict_in_generate=True, output_scores=True)
        else:
            response = model.module.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 16 + 4, return_dict_in_generate=True, output_scores=True)

        yes_prob, no_prob = response.scores[0][:,YES_TOKEN_IDS],response.scores[0][:,NO_TOKEN_IDS]
        logit = yes_prob/(yes_prob+no_prob)
        node_result = [(author[i],pub[i],logit[i].item()) for i in range(batch_size)]
        batch_result = accelerator.gather_for_metrics(node_result)
        if accelerator.is_main_process:
            result.extend(batch_result)

if accelerator.is_main_process: 
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    res_list = {}
    for i in result:
        [aid,pid,logit] = i
        if aid not in res_list.keys():
            res_list[aid] = {}
        res_list[aid][pid] = logit
    with open(f'{args.saved_dir}/result-{checkpoint}.json', 'w') as f:
        json.dump(res_list, f)