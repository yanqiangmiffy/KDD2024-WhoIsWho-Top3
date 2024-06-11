# KDD2024-WhoIsWho-Top3
KDD2024-WhoIsWho-Top3

## Prerequisites
基础环境：
```text
Linux
Python 3.11.7
PyTorch 2.2.0+cu121
```
requirements:
```text

lightgbm==4.3.0
pandas==1.5.3
numpy==1.24.4

```


## lgb codes
```text
│  step0_initialize.py:初始化路径
│  step1_load_data.py：将json数据转为csv格式
│  step2_train_text_models.py：训练文本模型：tfidf/count/word2vec/glove
│  step3_extract_features_emb.py:提取文本的嵌入特征
│  step3_extract_features_stats.py：提取作者以及相关paper的统计特征
│  step3_extract_oag_features.py：提取oag嵌入表示
│  step3_merge_features.py：合并所有的特征生成完整的训练集和测试集
│  step4_train_tree.py：训练lgb
│  utils.py:通用函数
│
├─cogdl
├─output
│  ├─step1
│  ├─step2
│  └─step3
└─result
```