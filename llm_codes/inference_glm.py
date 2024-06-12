# -*- coding: utf-8 -*-

import os
from peft import PeftModel,get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from utils.dataset import IND4EVAL
import json
from accelerate import Accelerator
from tqdm import tqdm
from utils.metric import compute_metric
import argparse
import torch.distributed as dist

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

model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half()
# if args.lora_path is not None:
#     tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
# else:
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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
    max_target_length = 128,
)
print('done reading dataset')

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids', 'author', 'pub')}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return batch_input,batch['author'],batch['pub']

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

        response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 16, return_dict_in_generate=True, output_scores=True)

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