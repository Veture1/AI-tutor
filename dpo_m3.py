#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
#from safetensors.torch import save_file, load_file
import torch.nn.utils.prune as prune

"""# 1. 加载 PyTorch 模型权重
model_weights_path = "./mnlp_sft_model/adapter_model.bin"
model_weights = torch.load(model_weights_path)

# 2. 保存为 safetensors 格式
safetensors_weights_path = "./mnlp_sft_model/adapter_model.safetensors"
save_file(model_weights, safetensors_weights_path)

print(f"Model weights saved to {safetensors_weights_path}")"""



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print device info
print("Using device:", device)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# load dataset
dataset = load_dataset("Tachi67/mnlp_dpo_data_7k")
train_dataset = dataset['train']
eval_dataset = dataset['test']

#################
# Experiment    #
#################
sample_size = 100
small_train_dataset = train_dataset.select(range(sample_size))
small_eval_dataset = dataset['test'].select(range(sample_size))


# 指定模型名称
model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# add padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# 加载 LoRA 适配器配置和权重
dpo_path = "./sft_test"
model = PeftModel.from_pretrained(model, dpo_path, is_trainable=True, dtype=torch.float16)



"""# 定义剪枝函数
def prune_lora_params(model, threshold=1e-5):
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            # 剪掉低于阈值的权重
            mask = torch.abs(param) > threshold
            param.data *= mask.float()

# 应用剪枝
prune_lora_params(model)"""

"""##############
# new lora   #
##############

#  LoRA settig
new_lora_config = LoraConfig(
    r=16,  # 低秩的维度
    lora_alpha=32,  # 缩放因子
    inference_mode=False,
    target_modules=[ "query_key_value"],  # 需要应用 LoRA 的层
    lora_dropout=0.1,  # Dropout 率
    bias="none",  # 是否在 LoRA 中适应偏置
    task_type="CAUSAL_LM"  # 任务类型
)"""


model.enable_input_require_grads()
#model.print_trainable_parameters()
model.to(device)

# set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="no",  # 每个 epoch 后进行评估
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=200,
    learning_rate=3e-4,
    weight_decay=0.05,
    logging_dir='./logs',
    logging_steps=50,
    fp16=True,
    fp16_opt_level='O1',
    remove_unused_columns=False,
    gradient_accumulation_steps=16,
)

# initialize Trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    #train_dataset=small_train_dataset,
    eval_dataset=eval_dataset,
    #eval_dataset=small_eval_dataset,
    max_length=512,
    max_prompt_length=512,
    #peft_config=adapter_config,
    #peft_config=new_lora_config
)

print(torch.cuda.memory_summary())
# 检查可训练参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

print("Part of the model weights：")
for name, param in model.named_parameters():
    if 'lora' in name:
        print(name, param.data)

torch.cuda.empty_cache()

# training
trainer.train()

# save model
model_output_dir = './saved_model'
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
model.save_pretrained("./best_model_dpo")


