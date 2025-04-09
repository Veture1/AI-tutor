import warnings
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from trl import SFTTrainer
from transformers import  AutoTokenizer, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EvalPrediction
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from tqdm import tqdm
import json
import numpy as np
from typing import List
import pandas as pd
import random
import os
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
tqdm.pandas()


if __name__ == "__main__":
    ##############
    # Model      #
    ##############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
        
    tokenizer.model_max_length = 512 # change to 1024 for full scale
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    print("Model on device:", device)
    # prepare model for LoRA
    # https://zhuanlan.zhihu.com/p/618073170
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R, # LoRA中低秩近似的秩
        lora_alpha=LORA_ALPHA, # 见上文中的低秩矩阵缩放超参数
        lora_dropout=LORA_DROPOUT, # LoRA层的dropout
        bias='none'
    )
    # dont do this. configure the peft stuff in the trainer.
    # # apply lora to model
    # model.enable_input_require_grads()
    # model = get_peft_model(model, lora_config)
    # model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))
    # model.print_trainable_parameters()
    
    ##############
    # Dataset    #
    ##############
    dataset = load_dataset("Tachi67/sft_dataset")
    # small dataset for testing
    training_sample = dataset['train']
    test_sample = dataset['test']


    # 定义格式化函数
    def formatting_prompts_func(example):
        formatted_texts = []
        for i in range(len(example['text'])):
            text = f"### {example['text'][i]}\n### Answer: {example['label'][i]}"
            formatted_texts.append(text)
        return formatted_texts
    
    
    #################
    # Experiment    #
    #################
    # small dataset
    # dataset_train = dataset['train'].train_test_split(test_size=0.998)['train']
    # dataset_val = dataset['test'].train_test_split(test_size=0.996)['train']
    # from datasets import DatasetDict
    # dataset_small = DatasetDict({
    #     'train': dataset_train,
    #     'test': dataset_val
    # })
    
    ##############
    # Training   #
    ##############
    training_args = TrainingArguments(
        output_dir="./results",          # 输出目录
        evaluation_strategy="no",     # 每个 epoch 后进行评估
        save_strategy="no",           # 每个 epoch 后保存模型
        learning_rate=1e-4,              # 学习率
        weight_decay=0.01,               # 权重衰减
        num_train_epochs=3,             # 训练轮数
        # gradient_checkpointing=True,     # 梯度检查点
        gradient_accumulation_steps=16, # 梯度累积
        # lr_scheduler_type="cosine",      # 学习率调度器
        logging_dir='./logs',
        logging_steps=10,
        optim="adamw_torch",         # 优化器
        per_device_train_batch_size=1,   # 每个设备的批量大小
        per_device_eval_batch_size=1,    # 每个设备的评估批量大小
        load_best_model_at_end=False,      # 训练结束时载入最佳模型
        save_total_limit=1,              # 最大保存模型数量
        fp16=True,                       # 混合精度训练
        fp16_opt_level='O1'              # 混合精度训练
    )
    
    trainer = SFTTrainer(
    model,
    train_dataset=training_sample,
    # train_dataset=dataset_small['train'],
    eval_dataset=test_sample,
    # eval_dataset=dataset_small['test'],
    formatting_func=formatting_prompts_func,
    args=training_args,
    #dataset_text_field="text",
    max_seq_length=512, # change to 1024 for full scale
    peft_config=lora_config,
    
)
    
    print(torch.cuda.memory_summary())

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)

    #print trainable parameters
    def count_trainable_parameters(model):
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable parameter: {name} - Size: {param.numel()}")

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")


    count_trainable_parameters(model)
    print("Training...")
    trainer.train()
    print("Completed!")
    print("Saving model...")
    trainer.save_model("sft_test")
    print("Pushing to hub...")
    trainer.model.push_to_hub("mnlp_sft_model_bloom")
    print("Done!")
