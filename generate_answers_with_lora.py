from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bigscience/bloom-1b7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# add padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# load LORA config
dpo_path = "./dpo_lora" #change this to switch the lora adapter
model = PeftModel.from_pretrained(model, dpo_path, dtype=torch.float16)
model.to(device)


# example question
question = "Question: Increasing the depth of a decision tree cannot increase its training error. Options: A. TRUE B. FALSE"
input_text = f"Question: {question}\nAnswer:"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

max_length = 512
output = model.generate(
    inputs["input_ids"],
    max_length=max_length,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
)

# generate answer
answer = tokenizer.decode(output[0], skip_special_tokens=True)
print(answer)


