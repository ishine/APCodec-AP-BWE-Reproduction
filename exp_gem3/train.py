import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
import json

def main():
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #路径参数
    MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
    TRAINING_FILE = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/train_data.json"
    VALIDATION_FILE = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/valid_data.json"
    OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_gem3"

    #加载tokenizer，扩展词表
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    new_special_tokens = {"additional_special_tokens": [f"<token_{i}>" for i in range(3072)] + ["<missing>"]}
    tokenizer.add_special_tokens(new_special_tokens)

    def create_dataset(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        formatted = []
        for item in raw_data:
            user_text = (
                "<start_of_turn>user\n"
                "Repair the text by filling in the <missing> tokens:\n"
                f"{item['input']}<end_of_turn>\n"
            )
            model_text = (
                "<start_of_turn>model\n"
                f"{item['output']}<end_of_turn><eos>"
            )
            full_text = user_text + model_text
            tokenized_text = tokenizer(full_text, truncation=True, max_length=2048)
            formatted.append(tokenized_text)
        
        return Dataset.from_list(formatted)

    # QLoRA和模型加载
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading base model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )

    #加载数据集
    print("Loading and formatting dataset...")
    train_dataset = create_dataset(TRAINING_FILE)
    eval_dataset = create_dataset(VALIDATION_FILE)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #训练参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=20,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        logging_steps=100,
        bf16=True,
        max_grad_norm=0.9,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    #resume_from_checkpoint="/mnt/nvme_share/srt30/checkpoint/exp_llm/checkpoint-55000"

    final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}")
    trainer.save_model(final_adapter_path)

if __name__ == '__main__':
    main()