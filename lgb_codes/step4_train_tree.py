#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/5/16 17:06
"""
# %%
import gc
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def load_data():
    train = pd.read_pickle('output/train.pkl')
    test = pd.read_pickle('output/test.pkl')
    print(train.shape, test.shape)

    train['author_year'] = train['author_id'].astype(str) + '_' + train['year'].astype(str)
    train['author_venue'] = train['author_id'].astype(str) + '_' + train['venue'].astype(str)
    test['author_year'] = test['author_id'].astype(str) + '_' + test['year'].astype(str)
    test['author_venue'] = test['author_id'].astype(str) + '_' + test['venue'].astype(str)
    train_year_oag = pd.read_pickle('output/step3/step3_train_paper_oag_author_year_text_features.pkl')
    test_year_oag = pd.read_pickle('output/step3/step3_test_paper_oag_author_year_text_features.pkl')
    train_venue_oag = pd.read_pickle('output/step3/step3_train_paper_oag_author_venue_text_features.pkl')
    test_venue_oag = pd.read_pickle('output/step3/step3_test_paper_oag_author_venue_text_features.pkl')
    train = train.merge(train_year_oag, how='left', on=['author_id', 'id', 'author_year'])
    test = test.merge(test_year_oag, how='left', on=['author_id', 'id', 'author_year'])
    train = train.merge(train_venue_oag, how='left', on=['author_id', 'id', 'author_venue'])
    test = test.merge(test_venue_oag, how='left', on=['author_id', 'id', 'author_venue'])
    del train_year_oag, test_year_oag, train_venue_oag, test_venue_oag
    del train['author_year'], train['author_venue'], test['author_year'], test['author_venue']

    # train['label']=train['label_x']
    # del train['label_y']
    # del train['label_x']
    print(train.shape, test.shape)
    print(train.columns)
    print(train['label'].value_counts())
    print(train.shape)
    # df = pd.read_pickle('output/all.pkl')
    # train = df[df['label'] != -1].reset_index(drop=True)
    # test = df[df['label'] == -1].reset_index(drop=True)
    # train=train[train['year'].isin(test['year'].unique())].reset_index(drop=True)
    # print(train.shape)

    train_sta = pd.read_pickle('H:/2024-Biendata-WhoIsWho/ml_codes/train.pkl')
    test_sta = pd.read_pickle('H:/2024-Biendata-WhoIsWho/ml_codes/test.pkl')
    feat = [i for i in train_sta.columns if
            i not in ['label', 'papers', 'title', 'name', 'authors', 'abstract', 'keywords', 'venue', 'full_text']]
    feat = [col for col in feat if col not in train.columns]

    train = pd.concat([train, train_sta[feat]], axis=1)
    test = pd.concat([test, test_sta[feat]], axis=1)
    with open("H:/2024-Biendata-WhoIsWho/data/IND-WhoIsWho/ind_test_author_submit.json") as f:
        submission = json.load(f)

    gc.collect()
    return train, test, submission


train, test, submission = load_data()
print(train.shape)

print(train.shape)

# lgb_params = {
#     # 'objective': 'binary',  # 二分类任务
#     'n_estimators': 3072,
#     'random_state': 42,
#     # 'learning_rate': 0.03,
#     'subsample': 0.6,
#     'subsample_freq': 1,
#     'colsample_bytree': 0.4,
#     'reg_alpha': 10.0,
#     'reg_lambda': 1e-1,
#     'min_child_weight': 256,
#     'min_child_samples': 500,
#     'verbose': 50,
#     'learning_rate': 0.01,
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'num_leaves': 63,
#     'feature_fraction': 0.6,
#     'bagging_fraction': 0.8,
#     'min_data_in_leaf': 15,
#     'nthread': 8,
#     'max_depth': 7,
#     'device': 'cpu'
# }
lgb_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 12,
    "learning_rate": 0.1,
    "n_estimators": 3000,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees": True,
    'num_leaves': 64,
    "verbose": -1,
    "max_bin": 255,
    # 'verbose': 50,
    'importance_type': 'gain'
}

lgb = LGBMClassifier(**lgb_params)


def get_oof(feats, target, test, kfold, clf, type='lgb'):
    oof_preds = np.zeros(feats.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    for i, (train_idx, valid_idx) in enumerate(kfold.split(feats, target, groups=train['author_id'])):
        train_X, train_y = feats.loc[train_idx], target.loc[train_idx]
        valid_X, valid_y = feats.loc[valid_idx], target.loc[valid_idx]
        if type == 'lgb':
            clf.fit(train_X, train_y, eval_set=[(valid_X, valid_y)],
                    callbacks=[log_evaluation(50), early_stopping(50)]
                    )
        else:
            clf.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], early_stopping_rounds=100,
                    )
        oof_preds[valid_idx] = clf.predict_proba(valid_X)[:, 1]
        sub_preds += clf.predict_proba(test)[:, 1]
        del train_X, train_y, valid_X, valid_y
        gc.collect()

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feature_names
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = i + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print(fold_importance_df.head(20))
        joblib.dump(clf,f"output/lgb0_fold{i}.joblib")

    fold_importance_df = feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(
        ascending=False).reset_index()
    evalution_result = roc_auc_score(target, oof_preds, average='weighted')

    print('*' * 10)
    print('roc auc score:', evalution_result)
    print('*' * 20)
    sub_preds_result = sub_preds / kfold.n_splits
    fold_importance_df.to_csv(f'output/{type}_importance_df_{evalution_result}.csv', index=False)
    return oof_preds, sub_preds_result, evalution_result


feature_names = [f for f in train.columns if f not in [
    'author_id', 'author_name', 'id', 'label', 'title', 'authors',
    'abstract', 'keywords', 'text', 'author_text', 'top_author',
    'first_author', 'second_author', 'first_author', 'top_words',
    'second_author', 'author_names', 'org_names', 'author_names_text',
    'org_names_text', 'top1_author', 'top2_author', 'top1_org', 'top2_org', 'venue', 'top1_keyword',
    'top2_keyword', 'author_title', 'author_keywords', 'author_abstract', 'author_author_names_text',
    'author_org_names_text', 'keywords_list', 'year'
    # 'no_splite_len_author_text','author_textkw_num','diff_no_sp','overlap_score',
    # 'author_text_nunique_text','author_text_count_text','no_splite_len_author_text'
]]

imp = pd.read_csv('output/lgb_importance_df_0.8552795230576107.csv')
extra_cols = [col for col in feature_names if col not in imp['feature'].values.tolist()]
print(extra_cols)
print(len(extra_cols))

feature_names = [col for col in feature_names if '_label_' not in col]
print(len(feature_names))
feature_names = [col for col in feature_names if 'jaccard' not in col]
print(len(feature_names))
feature_names = [col for col in feature_names if 'sorensen' not in col]

feature_names = [col for col in feature_names if col in imp['feature'].values.tolist()[:500]]
feature_names = list(set(feature_names))
# print(feature_names)
print('Current num of features:', len(feature_names))
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2023)
# kfold = GroupKFold(n_splits=5,)

lgb_oof_preds, lgb_preds, auc_score = get_oof(train[feature_names], train['label'], test[feature_names], kfold, lgb)

cnt = 0
for id, names in submission.items():
    for name in names:
        submission[id][name] = lgb_preds[cnt]
        cnt += 1
with open(f'result/{auc_score}_baseline_lgb.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)
