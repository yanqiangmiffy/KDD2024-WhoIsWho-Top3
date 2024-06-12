
set -ex

lora_path="your lora model path" #你训练好的lora模型所在的目录地址
path_to_model="your model path" #你的chatglm3_6b_32k模型所在目录地址

path_to_pub_file="path of pid_to_info_all.json" #pid_to_info_all.json文件路径
path_to_eval_author="path of ind_test_author_filter_public.json" #ind_test_author_filter_public.json文件路径
result_to_save=./result

set -o pipefail

accelerate launch --num_processes 8 inference_glm.py --lora_path $lora_path --model_path $path_to_model --pub_path $path_to_pub_file --eval_path $path_to_eval_author --saved_dir $result_to_save --batch_size 1 --max_source_length 8192

ls -lh $result_to_save