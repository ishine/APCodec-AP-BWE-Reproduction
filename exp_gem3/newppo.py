import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import PeftModel, LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import os
import json
import re
import glob
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "ground_truth_tokens": [f["ground_truth_tokens"] for f in features]
        }
        #对input_ids和attention_mask进行padding
        padded = self.tokenizer.pad(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            padding=True,
            return_tensors="pt"
        )
        batch.update(padded)
        return batch

class RewardModel(nn.Module):
    def __init__(self, tokenizer, n_quantizers=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_quantizers = n_quantizers

    def forward(self, input_ids, attention_mask=None, ground_truth_tokens=None):
        batch_size = input_ids.shape[0]
        rewards = torch.zeros(batch_size, device=input_ids.device)
        for i in range(batch_size):
            input_len = len(input_ids[i])
            new_ids = input_ids[i, input_len:]
            tokens = [int(m) for m in re.findall(r"<token_(\d+)>", self.tokenizer.decode(new_ids, skip_special_tokens=False))]
            gt = ground_truth_tokens[i] if ground_truth_tokens else []
            if len(tokens) != len(gt):
                rewards[i] = -100.0
                continue
            reward = 0.0
            for j, (r, g) in enumerate(zip(tokens, gt)):
                if j % self.n_quantizers == self.n_quantizers - 1: 
                    reward += 3.0 if r == g else 0.0 
                else:
                    reward += 1.0 if r == g else 0.0
            rewards[i] = reward / len(gt) if gt else 0.0
        return rewards

def prepare_dataset(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    formatted = []
    for item in raw_data:
        ground_truth = [int(m) for m in re.findall(r"<token_(\d+)>", item['output'])]
        user_text = (
            "<start_of_turn>user\n"
            "Repair the text by filling in the <missing> tokens:\n"
            f"{item['input']}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        tokenized = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=1024)
        
        formatted.append({
            "input_ids": tokenized.input_ids[0].tolist(),
            "attention_mask": tokenized.attention_mask[0].tolist(),
            "ground_truth_tokens": ground_truth
        })
    return Dataset.from_list(formatted)

def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
    SFT_ADAPTER_PATH = "/mnt/nvme_share/srt30/checkpoint/exp_gem3/checkpoint-18000"
    TRAINING_FILE = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/train_data.json"
    OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_gem3/ppo"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    new_special_tokens = {"additional_special_tokens": [f"<token_{i}>" for i in range(1024)] + ["<missing>"]}
    tokenizer.add_special_tokens(new_special_tokens)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Policy Model
    logging.info(f"Loading base model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
    model.train()
    print(f"Policy model loaded on: {set(p.device for p in model.parameters())}")

    # Reference Model
    logging.info(f"Loading reference model from: {MODEL_PATH}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model = PeftModel.from_pretrained(ref_model, SFT_ADAPTER_PATH)
    ref_model.eval()
    print(f"Ref model loaded on: {set(p.device for p in ref_model.parameters())}")

    # Value Model
    value_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    value_model.resize_token_embeddings(len(tokenizer))
    value_model = PeftModel.from_pretrained(value_model, SFT_ADAPTER_PATH)
    value_model.train()
    print(f"Value model loaded on: {set(p.device for p in value_model.parameters())}")

    # Reward Model
    reward_model = RewardModel(tokenizer=tokenizer, n_quantizers=3).to(device)

    # Load Dataset
    logging.info("Loading and formatting dataset...")
    train_dataset = prepare_dataset(TRAINING_FILE, tokenizer)

    # PPO 配置
    ppo_config = PPOConfig(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
        learning_rate=1e-6,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        save_strategy="steps",
        save_steps=3000,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # PPO Trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=value_model,
        data_collator=data_collator,
    )

    # Train
    logging.info("Starting PPO training...")
    ppo_trainer.train()

    # Save 
    final_ppo_path = os.path.join(OUTPUT_DIR, "final_ppo_adapter")
    logging.info(f"Saving final PPO adapter to {final_ppo_path}")
    ppo_trainer.save_model(final_ppo_path)

if __name__ == '__main__':
    main()