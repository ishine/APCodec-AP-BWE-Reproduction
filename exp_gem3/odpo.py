import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from trl import OnlineDPOConfig, OnlineDPOTrainer, BasePairwiseJudge
import os
import json
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def prepare_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    gt_map = {}
    data = []
    for item in raw_data:
        gt = [int(m) for m in re.findall(r"<token_(\d+)>", item["output"])]
        prompt = (
            "<start_of_turn>user\n"
            "Repair the text by filling in the <missing> tokens:\n"
            f"{item['input']}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        gt_map[prompt] = gt
        data.append({"prompt": prompt})

    return Dataset.from_list(data), gt_map

class TokenMatchJudge(BasePairwiseJudge):

    def __init__(self, gt_map: dict, n_quantizers: int = 3):
        super().__init__()
        self.gt_map = gt_map
        self.n_quantizers = n_quantizers

    def judge(self, prompts, completions, **kwargs):

        results = []
        for prompt, pair in zip(prompts, completions):
            gt = self.gt_map.get(prompt, [])
            if not gt:
                results.append(0)
                continue
            print("DEBUG pair0:", pair[0][:120])
            print("DEBUG pair1:", pair[1][:120])
            if pair[0] == pair[1]:
                print("Warning: Received identical completions!")

            def score(tokens):
                if not tokens:
                    return -1.0 
                
                total = 0.0
                min_len = min(len(tokens), len(gt))
                for i in range(min_len):
                    r, g = tokens[i], gt[i]
                    if (i % self.n_quantizers) == (self.n_quantizers - 1):
                        total += 3.0 if r == g else 0.0
                    else:
                        total += 1.0 if r == g else 0.0
            
                length_penalty = abs(len(tokens) - len(gt))

                return (total / len(gt)) - (length_penalty * 0.1)

            toks_a = [int(m) for m in re.findall(r"<token_(\d+)>", pair[0])]
            toks_b = [int(m) for m in re.findall(r"<token_(\d+)>", pair[1])]
            
            sa, sb = score(toks_a), score(toks_b)
            results.append(0 if sa >= sb else 1)
            print(f"results:{results[-1]}, scores: {sa:.3f} vs {sb:.3f}")

        return results

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
    SFT_ADAPTER_PATH = "/mnt/nvme_share/srt30/checkpoint/exp_gem3/checkpoint-18000"
    TRAINING_FILE = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_gem3/train_data.json"
    OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_gem3/odpo"

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
    #load dataset
    train_dataset, gt_map = prepare_dataset(TRAINING_FILE)

    #judge
    judge = TokenMatchJudge(gt_map, n_quantizers=3)

    # load policy model
    logging.info(f"Loading policy model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
    model.train()
    model.config.use_cache = False

    # load reference model
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
    ref_model.config.use_cache = False

    training_args = OnlineDPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        report_to="tensorboard",
        max_length=2048,
        beta=0.1,
        missing_eos_penalty=1.0,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
    )

    trainer = OnlineDPOTrainer(
        model=model,
        ref_model=ref_model,
        judge=judge,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable)} / total {len(list(model.parameters()))}")

    logging.info("Starting OnlineDPO training...")
    trainer.train()

    logging.info("Save final adapter to :", OUTPUT_DIR)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_odpo_adapter"))

if __name__ == "__main__":
    main()