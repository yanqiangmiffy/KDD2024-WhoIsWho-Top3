import logging
import torch
import os
import sys
import json
import transformers
import bitsandbytes as bnb
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig
)
from transformers import Trainer, deepspeed, set_seed
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from utils.dataset import (
    INDDataSetForGLM, 
    INDDataSetForMistral, 
    INDDataSetForGLM4
)
from utils.collator import DataCollatorForGLM, DataCollatorForLLama
from utils.trainer import LoRATrainer
from utils.arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from trl import get_kbit_device_map
import importlib
if importlib.util.find_spec('unsloth') is not None:
    from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

def main(): 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.ddp_find_unused_parameters = False
    training_args.save_safetensors = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    print(f'tokenizer special_tokens_map: {tokenizer.special_tokens_map}')
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"

    logger.info('Train model with sft task')
    with open(data_args.pub_data, "r" , encoding = "utf-8") as f:
        pub_data = json.load(f)
    with open(data_args.train_data, "r", encoding="utf-8") as f: 
        train_data = json.load(f)

    if data_args.model_type == 'glm':
        train_dataset = INDDataSetForGLM(
            (train_data,pub_data),
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
    elif data_args.model_type == 'casual':
        # train_dataset = INDDataSetForMistral(
        #     (train_data,pub_data),
        #     tokenizer,
        #     data_args.max_source_length,
        #     data_args.max_target_length,
        # )
        train_dataset = INDDataSetForGLM4(
            (train_data,pub_data),
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
        training_args.remove_unused_columns = False
    else:
        raise ValueError("model_type cannot be None")
    
    max_seq_len = train_dataset.max_seq_len

    def print_supervised_dataset_example(example) -> None:
        IGNORE_INDEX = -100
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        if 'target_mask' in example:
            target_ids = []
            for i, j in zip(example["input_ids"], example["target_mask"]):
                if int(j) == 1:
                    target_ids.append(i)
            print("label_ids:\n{}".format(target_ids))
            print("labels:\n{}".format(tokenizer.decode(target_ids), skip_special_tokens=False))
        elif 'labels' in example:
            print("label_ids:\n{}".format(example["labels"]))
            print("labels:\n{}".format(tokenizer.decode(example["labels"]), skip_special_tokens=False))
        else:
            pass

    if training_args.local_rank == 0:
        print_supervised_dataset_example(next(iter(train_dataset)))

    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {model_args.model_name_or_path}')
    logger.info(f'Train model with {data_args.model_type}')

    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16

    if data_args.model_type == 'glm':
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.use_cache = False
        if data_args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=False,)

            model = AutoModel.from_pretrained(model_args.model_name_or_path,
                                              torch_dtype=torch_dtype,
                                              quantization_config=quantization_config,
                                              trust_remote_code=True)

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        else:
            model = AutoModel.from_pretrained(model_args.model_name_or_path, 
                                              torch_dtype=torch_dtype,
                                              config=config, 
                                              trust_remote_code=True)

        print(f'vocab_size of model: {model.config.vocab_size}')
        if model.config.vocab_size < tokenizer.vocab_size:
            print(f'Change vocab_size of model: {model.config.vocab_size} -> {tokenizer.vocab_size}')
            model.resize_token_embeddings(tokenizer.vocab_size)

        if data_args.use_lora or data_args.use_qlora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=['query_key_value'],
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
            )

            logger.info('Using lora or qlora...')
            model = get_peft_model(model, peft_config)
            logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
            model.print_trainable_parameters()

    elif data_args.model_type == 'casual':
        if not data_args.use_unsloth:
            if data_args.use_qlora:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            else:
                quantization_config = None
            model_kwargs = dict(
                trust_remote_code=True,
                # attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                use_cache=False if training_args.gradient_checkpointing else True,
                device_map=get_kbit_device_map() if quantization_config is not None else None,
                quantization_config=quantization_config,
            )
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

            # 需要扩词表
            print(f'vocab_size of model: {model.config.vocab_size}')
            if model.config.vocab_size < tokenizer.vocab_size:
                print(f'Change vocab_size of model: {model.config.vocab_size} -> {tokenizer.vocab_size}')
                model.resize_token_embeddings(tokenizer.vocab_size)

            # moe模型，需要考虑负载均衡的loss
            if 'output_router_logits' in model.config.to_dict():
                logger.info('set output_router_logits as True')
                model.config.output_router_logits = True
            
            if data_args.use_qlora:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

            if data_args.use_lora:
                # For backward compatibility
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
            # init peft_config
            if data_args.use_lora or data_args.use_qlora:
                # 找到所有需要插入adapter的全连接层
                target_modules = ['query_key_value']
                peft_config = LoraConfig(
                    r=model_args.lora_rank,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            else:
                peft_config = None

            if data_args.use_lora or data_args.use_qlora:
                logger.info('Using lora or qlora...')
                model = get_peft_model(model, peft_config)
                logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
                model.print_trainable_parameters()
        else:
            logger.info('Using unsloth...')
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_args.model_name_or_path,
                max_seq_length=max_seq_len,
                dtype=None,
                trust_remote_code=True,
                load_in_4bit=True if data_args.use_qlora else False,
            )
            if data_args.use_lora or data_args.use_qlora:
                logger.info('Initializing PEFT Model...')
                target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'c_attn']
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=model_args.lora_rank,
                    target_modules=target_modules,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=training_args.seed,
                    max_seq_length=max_seq_len,
                )
                logger.info(f'target_modules: {target_modules}')
    else:
        raise ValueError("model_type cannot be None")
    
    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fB" % (total / 1e9))

    if data_args.model_type == 'glm':
        data_collator = DataCollatorForGLM(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False,
        )
    elif data_args.model_type == 'casual':
        data_collator = DataCollatorForLLama(tokenizer=tokenizer, max_seq_len=max_seq_len)
    else:
        raise ValueError("model_type cannot be None")

    if data_args.use_lora or data_args.use_qlora:
        trainer = LoRATrainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    logger.info("*** starting training ***")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)

    if data_args.use_lora or data_args.use_qlora:
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
    else:
        trainer.save_model(training_args.output_dir)
        trainer.model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()