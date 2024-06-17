#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/13 22:30
"""

import json
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd

num_folds = 5
with open("../data/IND-WhoIsWho/ind_test_author_submit.json") as f:
    submission = json.load(f)
print(len(submission))


test0 = pd.read_pickle('output/test0.pkl')
sub_preds0 = np.zeros(test0.shape[0])
for fold in tqdm(range(num_folds)):
    clf = joblib.load(f'output/lgb0_fold{fold}.joblib')
    sub_preds0 += clf.predict_proba(test0)[:, 1]
lgb_preds0 = sub_preds0 / num_folds
cnt = 0
for id, names in submission.items():
    for name in names:
        submission[id][name] = lgb_preds0[cnt]
        cnt += 1
with open(f'result/lgb0.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)



test1 = pd.read_pickle('output/test1.pkl')
sub_preds1 = np.zeros(test1.shape[0])
for fold in tqdm(range(num_folds)):
    clf = joblib.load(f'output/lgb1_fold{fold}.joblib')
    sub_preds1 += clf.predict_proba(test1)[:, 1]
lgb_preds0 = sub_preds1 / num_folds
cnt = 0
for id, names in submission.items():
    for name in names:
        submission[id][name] = lgb_preds0[cnt]
        cnt += 1
with open(f'result/lgb1.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)



test2 = pd.read_pickle('output/test2.pkl')
sub_preds2 = np.zeros(test2.shape[0])
for fold in tqdm(range(num_folds)):
    clf = joblib.load(f'output/lgb2_fold{fold}.joblib')
    sub_preds2 += clf.predict_proba(test2)[:, 1]
lgb_preds2 = sub_preds2 / num_folds
cnt = 0
for id, names in submission.items():
    for name in names:
        submission[id][name] = lgb_preds2[cnt]
        cnt += 1
with open(f'result/lgb2.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)
