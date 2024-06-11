# KDD2024-WhoIsWho-Top3
KDD2024-WhoIsWho-Top3

## 1 运行环境
基础环境：
```text
Linux
Python 3.11.7
```
requirements:
```text
torch==2.3.0
torchaudio==2.3.0+cu121
torchvision==0.18.0+cu121
lightgbm==4.3.0
pandas==1.5.3
numpy==1.24.4
transformers==4.40.1
sentence-transformers==2.7.0
scikit-learn==1.2.2
gensim==4.3.0
```


## 2 lgb codes

### 2.1 文件说明
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
├─cogdl：oag-bert
├─output
│  ├─step1
│  ├─step2
│  └─step3
└─result
```

### 2.2 建模思路
构建作者画像以及与paper的差异性特征，主要通过聚合文本来表示author，然后计算author的文本表示与每个paper的文本表示的距离特征，文本字段主要有论文的title/abstract/keywords

### 2.3 实验总结
- 特征重要性如下:

> 通过下面我们可以发下通过计算作者向量(尤其oag的表示)与每个paper的向量的距离特征很重要，另外每个作者的发表论文数量以及对应组织、合作第一作者出现统计特征也比较重要

![imp](resources/imp.png)

- 通过不同的Embedding表示构建的特征维度很大，最后通过特征重要性筛选，发现取重要特征top500的效果最好
- StratifiedKFold、StratifiedGroupKFold、GroupKFold划分方式效果差不多，其中`StratifiedGroupKFold`稍微高一些
- lgb、xgb、cat融合效果没有提升，lgb效果最优
- 作者历史发表论文与待测论文的文本交叉特征比较重要


## 3 llm_codes

### 3.1 建模思路

### 3.2 实验总结

## 团队介绍
- **asir** B站大模型团队 算法工程师
- **yanqiang** 中科院计算所GoMate团队 算法工程师
