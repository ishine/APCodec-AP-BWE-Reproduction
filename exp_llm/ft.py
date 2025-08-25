import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import Accelerator
import os

MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
DATASET_PATH = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_llm/train_data.json"
OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_new_llm" 

INSTRUCTION_PROMPT = "Repair the following token sequence:"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def create_chat_format(example):

    dropped_str = ' '.join(map(str, example['dropped_tokens']))
    original_str = ' '.join(map(str, example['original_tokens']))
    
    messages = [
        {"role": "user", "content": f"{INSTRUCTION_PROMPT}\n{dropped_str}"},
        {"role": "assistant", "content": original_str}
    ]

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

print("Loading and formatting dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
formatted_dataset = dataset.map(create_chat_format)
print(f"Dataset formatted. First example:\n{formatted_dataset[0]['text']}")


# --- QLoRA 和模型加载 ---
# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载量化后的模型
print(f"Loading base model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={'': Accelerator().process_index},
    trust_remote_code=True,
)

# LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=500,
    save_strategy="steps",
    save_steps=5000,
    logging_steps=500,
    bf16=True,
    #max_grad_norm=0.9,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    group_by_length=True,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=training_args,
)

print("Starting training...")
trainer.train(resume_from_checkpoint="/mnt/nvme_share/srt30/checkpoint/exp_new_llm/checkpoint-550000")
#resume_from_checkpoint="/mnt/nvme_share/srt30/checkpoint/exp_llm/checkpoint-55000"

final_model_path = os.path.join(OUTPUT_DIR, "final_adapter")
print(f"Saving final LoRA adapter to {final_model_path}")
trainer.save_model(final_model_path)

print("Training complete!")
print(f"Your fine-tuned LoRA adapter is saved at: {final_model_path}")