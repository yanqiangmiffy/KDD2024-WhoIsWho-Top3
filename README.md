# KDD2024-WhoIsWho-Top3

## 1 Environment
Base environment:
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


## 2 LGB Codes

### 2.1 File Description
```text
│  step0_initialize.py: Initialize paths
│  step1_load_data.py: Convert JSON data to CSV format
│  step2_train_text_models.py: Train text models: tfidf/count/word2vec/glove
│  step3_extract_features_emb.py: Extract text embedding features
│  step3_extract_features_stats.py: Extract statistical features of authors and related papers
│  step3_extract_oag_features.py: Extract oag embedding representation
│  step3_merge_features.py: Merge all features to create a complete training and test dataset
│  step4_train_tree.py: Train lgb1
│  step4_train_tree_imp.py: Train lgb2 with imp+oag features
│  utils.py: Utility functions
│
├─cogdl: oag-bert
├─output
│  ├─step1
│  ├─step2
│  └─step3
└─result
```
Run the following steps:

```shell
python step0_initialize.py
python step1_load_data.py
python step2_train_text_models.py
python step3_extract_features_emb.py
python step3_extract_features_stats.py
python step3_extract_oag_features.py
python step3_merge_features.py
python step4_train_tree.py lgb1
python step4_train_tree_imp.py: imp+oag features lgb2
```

Training weights link:
```text
Link: https://pan.baidu.com/s/15t9Be4LS7NK5bnRETnZjoQ?pwd=svis 
Code: svis 
--Shared by a Baidu Netdisk Super VIP V5
```

### 2.2 Modeling Approach
Building author profiles and features to differentiate them from papers, primarily by aggregating text to represent the author, then calculating the distance features between the author's text representation and each paper's text representation. Main text fields include the paper's title, abstract, and keywords.

![](resources/WhoIsWho_01.png)

### 2.3 Experiment Summary
- Feature importance is as follows:

> From the below, we can see that calculating the distance features between the author vector (especially oag representation) and each paper vector is very important, as well as the count of papers published by each author and the statistical features of corresponding organizations and co-authors.

![imp](resources/imp.png)

- A large number of features were constructed using different Embedding representations, and after filtering by feature importance, taking the top 500 most important features proved to be the best approach.
- StratifiedKFold, StratifiedGroupKFold, and GroupKFold partition methods had similar effects, with `StratifiedGroupKFold` being slightly higher.
- The fusion of lgb, xgb, and cat did not improve results; lgb was the best performing model.
- Textual cross-features between the author's historical published papers and the test paper were particularly important.


## 3 LLM Codes
### 3.1 File Description
```text
│  .gitkeep
│  finetune_chatglm3.py: chatglm3-6b fine-tuning code
│  finetune_glm4.py: glm4-9b fine-tuning code
│  finetune_mistral.py: Mistral-7b fine-tuning code
│  inference_for_mistral.py: Mistral-7b inference code
│  inference_glm.py: chatglm3-6b inference code
│  inference_glm4.py: glm4-9b inference code
│  infer_chatglm3.sh: chatglm3-6b inference script
│  infer_glm4.sh: glm4-9b inference script
│  infer_mistral.sh: Mistral-7b inference script
│  README.md
│  train_chatglm3.sh: chatglm3-6b training script
│  train_glm4.sh: glm4-9b training script
│  train_mistral.sh: Mistral-7b training script
│
├─configs
│      ds_config_zero2.json: Deepspeed configuration file
│
└─utils
        arguments.py: Arguments
        collator.py: Data builder
        dataset.py: Dataset
        trainer.py: Trainer

```

### 3.2 Environment
- GPU：8xA800 80g
- python：python3.10.12
- torch：2.1.0a0+4136153
- torchvision：0.16.0a0

The versions of other installation packages will be automatically installed when the training file is executed. Please make sure that your environment supports deepspeed training.

### 3.3 Open Source Model
We will train the following three models.
- chatglm3_6b_32k: https://huggingface.co/THUDM/chatglm3-6b-32k
- Mistral-7B-Instruct-v0.2: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- glm-4-9b-chat: https://huggingface.co/THUDM/glm-4-9b-chat

### 3.4 Run

All the following script commands are executed in the **llm_codes** directory.

First, create a folder to save the prediction results.
```
mkdir -p ./base_model
mkdir -p ./result
```

- Lora weight file link:

The competition's trained lora weights are stored in the provided cloud storage. If you wish to infer directly without training, please download and place them in the **base_model** directory.
```
Link: https://pan.baidu.com/s/1dCkVsYkde-j1aLODQwyG4Q 
Code: 3923
```

- chatglm3-6b

To train chatglm3, you need to modify the following variables in train_chatglm3.sh and then execute **bash train_chatglm3.sh** 
```
BASE_MODEL_PATH="your model path" # Directory address of your chatglm3_6b_32k model
PUB_PATH="path of pid_to_info_all.json" # Path to the pid_to_info_all.json file
TRAIN_PATH="path of train_author.json" # Path to the train_author.json file
```
To infer chatglm3, you need to modify the following variables in infer_chatglm3.sh and then execute **bash infer_chatglm3.sh**
```
lora_path="your lora model path" # Directory address of your trained lora model
path_to_model="your model path" # Directory address of your chatglm3_6b_32k model

path_to_pub_file="path of pid_to_info_all.json" # Path to the pid_to_info_all.json file
path_to_eval_author="path of ind_test_author_filter_public.json" # Path to the ind_test_author_filter_public.json file
```

- mistral-7b

To train Mistral, you need to modify the following variables in train_mistral.sh and then execute **bash train_mistral.sh** 

```
BASE_MODEL_PATH="your model path" # Directory address of your Mistral-7B-Instruct-v0.2 model
PUB_PATH="path of pid_to_info_all.json" # Path to the pid_to_info_all.json file
TRAIN_PATH="path of train_author.json" # Path to the train_author.json file
```
To infer Mistral, you need to modify the following variables in infer_mistral.sh and then execute **bash infer_mistral.sh**
```
lora_path="your lora model path" # Directory address of your trained lora model
path_to_model="your model path" # Directory address of your Mistral-7B-Instruct-v0.2 model

path_to_pub_file="path of pid_to_info_all.json" # Path to the pid_to_info_all.json file
path_to_eval_author="path of ind_test_author_filter_public.json" # Path to the ind_test_author_filter_public.json file
```

- glm4-9b

To train glm4, you need to modify the following variables in train_glm4.sh and then execute **bash train_glm4.sh** 

```
BASE_MODEL_PATH="your model path" # Directory address of your glm-4-9b-chat model
PUB_PATH="path of pid_to_info_all.json" # Path to the pid_to_info_all.json file
TRAIN_PATH="path of train_author.json" # Path to the train_author.json file
```
To infer glm4, you need to modify the following variables in infer_glm4.sh and then execute **bash infer_glm4.sh**
```
lora_path="your lora model path" # Directory address of your trained lora model
path_to_model="your model path"  # Directory address of your glm-4-9b-chat model

path_to_pub_file="path of pid_to_info_all.json" # Path to the pid_to_info_all.json file
path_to_eval_author="path of ind_test_author_filter_public.json" # Path to the ind_test_author_filter_public.json file
```

After running, check if there are three prediction result files in your result folder.

### 3.5 Modeling Approach

Using large models to determine if a specific text (the "Target Paper") belongs to a given set of author texts (the "Paper Collection").

- `Context Papers` are the collection of papers belonging to the current author.
- `Target Paper` is the paper to be tested.

![instruction.png](resources/instruction.png)

### 3.6 Experiment Summary

Fine-tuning ChatGLM3, GLM4-Chat, and Mistral-7B models using Lora, then obtaining corresponding result files for fusion.

## 4 Model Fusion
Normalization and weighted merging of multiple prediction results, see the `merge.py` file:

First, define the normalization and merging functions, read the prediction results from JSON files, normalize the read results, and then merge the normalized results according to given weights, saving the merged results to a new JSON file.

## 5 Team Introduction
- **Asir** - Algorithm Engineer from the Bilibili Big Model Team
- **yanqiang** - Algorithm Engineer from the CAS Institute of Computing Technology GoMate Team
