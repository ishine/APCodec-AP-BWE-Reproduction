import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score
import os
import json

MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
ADAPTER_PATH = "/mnt/nvme_share/srt30/checkpoint/exp_new_llm/checkpoint-80000"
TEST_DATA_PATH = "/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_llm/test_data.json"
OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_llm"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions_49k.json")
INSTRUCTION_PROMPT = "Repair the following token sequence:"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading base model from {MODEL_PATH}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config = bnb_config,
    device_map = {"":0},
    trust_remote_code = True
)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)

model = model.to("cuda:0")

test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
print(f"Load {len(test_dataset)} test examples")

def prediction(dataset, max_new_tokens=512):
    predictions = []
    ground_truths = []
    results = []
    
    for example in dataset:
        dropped_str = ' '.join(map(str, example['dropped_tokens']))
        ground_truth = example['original_tokens']
        # 将整数转换为字符串后再join
        gt_str = ' '.join(str(token) for token in ground_truth)
        gt_token_len = len(tokenizer(gt_str, add_special_tokens=False)["input_ids"])
        ground_truths.append(ground_truth)
        chat = [
            {"role": "user", "content": f"{INSTRUCTION_PROMPT}\n{dropped_str}"}
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        print(f"prompt:{prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        print("predicting tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gt_token_len+5,
                do_sample=False,  # 贪婪解码
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[-1]
        predicted_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"predicted_text:{predicted_text}")
        print("print predicted_text finished")
        predicted_tokens = list(map(int, predicted_text.strip().split()))
        predictions.append(predicted_tokens)
        results.append({"predicted_tokens": predicted_tokens})
        print(f"predicted_tokens:{predicted_tokens}\nlength:{len(predicted_tokens)}")
        print(f"ground_truth:{ground_truth}\nlength:{len(ground_truth)}")

    with open(PREDICTIONS_PATH, 'w') as f:
        f.write('[\n')
        for item in results:
            f.write(json.dumps(item, separators=(',', ':')) + ',\n')
        f.write(']\n')
    print(f"Predictions saved to: {PREDICTIONS_PATH}")

    return predictions, ground_truths

print("Generating predictions...")
predictions, ground_truths = prediction(test_dataset)