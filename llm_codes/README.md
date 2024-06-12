# LLM程序说明

## 1. 运行环境
- GPU：8xA800 80g
- python：python3.10.12
- torch：2.1.0a0+4136153
- torchvision：0.16.0a0

其余安装包的版本在训练sh文件中均会在执行时自动安装，请确保你的环境支持deepspeed训练

待用到的模型地址
- chatglm3_6b_32k: https://huggingface.co/THUDM/chatglm3-6b-32k
- Mistral-7B-Instruct-v0.2: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- glm-4-9b-chat: https://huggingface.co/THUDM/glm-4-9b-chat

## 2. 文件目录


## 3. 运行说明
以下脚本命令均在当前目录下执行

首先创建一个预测结果保存的文件夹
```
mkdir -p ./result
```
#### chatglm3-6b
训练chatglm3，你需要自行修改train_chatglm3.sh中以下变量后执行**bash train_chatglm3.sh** 
```
BASE_MODEL_PATH="your model path" #你的chatglm3_6b_32k模型所在目录地址
PUB_PATH="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
TRAIN_PATH="path of train_author.json" #train_author.json文件路径
```
推理chatglm3，你需要自行修改infer_chatglm3.sh中一下变量后执行**bash infer_chatglm3.sh**
```
lora_path="your lora model path" #你训练好的lora模型所在的目录地址
path_to_model="your model path" #你的chatglm3_6b_32k模型所在目录地址

path_to_pub_file="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
path_to_eval_author="path of ind_test_author_filter_public.json" #ind_test_author_filter_public.json文件路径
```

#### mistral-7b
训练mistral，你需要自行修改train_mistral.sh中以下变量后执行**bash train_mistral.sh** 
```
BASE_MODEL_PATH="your model path" #你的Mistral-7B-Instruct-v0.2模型所在目录地址
PUB_PATH="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
TRAIN_PATH="path of train_author.json" #train_author.json文件路径
```
推理mistral，你需要自行修改infer_mistral.sh中一下变量后执行**bash infer_mistral.sh**
```
lora_path="your lora model path" #你训练好的lora模型所在的目录地址
path_to_model="your model path" #你的Mistral-7B-Instruct-v0.2模型所在目录地址

path_to_pub_file="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
path_to_eval_author="path of ind_test_author_filter_public.json" #ind_test_author_filter_public.json文件路径
```

#### glm4-9b
训练glm4，你需要自行修改train_glm4.sh中以下变量后执行**bash train_glm4.sh** 
```
BASE_MODEL_PATH="your model path" #你的glm-4-9b-chat模型所在目录地址
PUB_PATH="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
TRAIN_PATH="path of train_author.json" #train_author.json文件路径
```
推理glm4，你需要自行修改infer_glm4.sh中一下变量后执行**bash infer_glm4.sh**
```
lora_path="your lora model path" #你训练好的lora模型所在的目录地址
path_to_model="your model path"  #你的glm-4-9b-chat模型所在目录地址

path_to_pub_file="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
path_to_eval_author="path of ind_test_author_filter_public.json" #ind_test_author_filter_public.json文件路径
```

运行结束后检查你的result文件夹下是否有3个预测结果文件