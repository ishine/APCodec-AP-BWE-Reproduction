import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig
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

def manage_checkpoints(output_dir, max_checkpoints=2):

    checkpoint_dirs = glob.glob(os.path.join(output_dir, "ppo_checkpoint_*"))
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda x: int(re.search(r"ppo_checkpoint_(\d+)", x).group(1)),
        reverse=True
    )
    if len(checkpoint_dirs) > max_checkpoints:
        for old_checkpoint in checkpoint_dirs[max_checkpoints:]:
            shutil.rmtree(old_checkpoint, ignore_errors=True)
            logging.info(f"Deleted old checkpoint: {old_checkpoint}")

class RewardModel:
    def __init__(self, n_quantizers=3):
        self.n_quantizers = n_quantizers

    def compute_reward(self, repaired_tokens, ground_truth_tokens):

        if len(repaired_tokens) != len(ground_truth_tokens):
            return -100.0
        
        reward = 0.0
        for i, (r, g) in enumerate(zip(repaired_tokens, ground_truth_tokens)):
            if i % self.n_quantizers == self.n_quantizers - 1: 
                reward += 3.0 if r == g else 0.0 
            else:
                reward += 1.0 if r == g else 0.0
        reward = reward / len(ground_truth_tokens)
        
        return reward

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
        tokenized = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=2048)
        
        formatted.append({
            "query": tokenized.input_ids[0].tolist(),
            "attention_mask": tokenized.attention_mask[0].tolist(),
            "ground_truth_tokens": ground_truth
        })

    return Dataset.from_list(formatted)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def main():

    MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
    SFT_ADAPTER_PATH = "/mnt/nvme_share/srt30/checkpoint/exp_gem3/checkpoint-18000"
    TRAINING_FILE = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/train_data.json"
    OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_gem3/ppo"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    new_special_tokens = {"additional_special_tokens": [f"<token_{i}>" for i in range(1024)] + ["<missing>"]}
    tokenizer.add_special_tokens(new_special_tokens)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logging.info(f"Loading base model from: {MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model.train()

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

    # 加载数据集
    logging.info("Loading and formatting dataset...")
    train_dataset = prepare_dataset(TRAINING_FILE, tokenizer)

    # PPO 配置
    ppo_config = PPOConfig(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        report_to="tensorboard",
        learning_rate=1e-6,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        deepspeed="deepspeed_zero3.yaml",
    )

    # 初始化奖励模型
    reward_model = RewardModel(n_quantizers=3)

    # PPO 训练器
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        processing_class = tokenizer,
        data_collator = collator,
    )

    # PPO 训练循环
    logging.info("Starting PPO training...")
    for step, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["query"]
        ground_truth_tokens = batch["ground_truth_tokens"]

        # 生成响应
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=1024,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 计算奖励
        rewards = []
        for i, r in enumerate(response_tensors):
            input_len = len(query_tensors[i])
            new_ids = r[input_len:]
            generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
            tokens = [int(m) for m in re.findall(r"<token_(\d+)>", generated_text)]
            gt_tokens = ground_truth_tokens[i]
            reward = reward_model.compute_reward(tokens, gt_tokens)
            rewards.append(torch.tensor(reward, device=device))

        # PPO 优化
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # 记录日志
        if step % 100 == 0:
            logging.info(f"Step {step}: {stats}")

        # 保存模型
        if step % 3000 == 0 and step > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"ppo_checkpoint_{step}")
            ppo_trainer.save_model(checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            manage_checkpoints(OUTPUT_DIR, max_checkpoints=2)

    # 保存最终模型
    final_ppo_path = os.path.join(OUTPUT_DIR, "final_ppo_adapter")
    logging.info(f"Saving final PPO adapter to {final_ppo_path}")
    ppo_trainer.save_model(final_ppo_path)

if __name__ == '__main__':
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()