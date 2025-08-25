import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score
import os
import json
import numpy as np
from exp_fsq.utils import AttrDict
from exp_fsq.dataset import amp_pha_specturm, load_wav
from exp_fsq.models import Encoder, Decoder
import logging
import soundfile as sf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    logging.info(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    logging.info("Complete.")
    return checkpoint_dict

MODEL_PATH = "/mnt/nvme_share/common/LLMs/gemma-2b-it"
ADAPTER_PATH = "/mnt/nvme_share/srt30/checkpoint/exp_new_llm/checkpoint-310000"
TEST_DATA_PATH = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_llm/test.json"
OUTPUT_DIR = "/mnt/nvme_share/srt30/checkpoint/exp_llm"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions_80k.json")
INSTRUCTION_PROMPT = "Repair the following token sequence:"

config_file = "/mnt/nvme_share/srt30/APCodec-Reproduction/exp_fsq/config.json"
with open(config_file, 'r') as f:
    json_config = json.load(f)
h = AttrDict(json_config)

if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

encoder = Encoder(h).to(device)
decoder = Decoder(h).to(device)

state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
encoder.load_state_dict(state_dict_encoder['encoder'])
state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
decoder.load_state_dict(state_dict_decoder['decoder'])

encoder.eval()
decoder.eval()

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

def prediction(dataset, max_new_tokens=512):
    predictions = []
    ground_truths = []
    results = []
    total_tokens = []
    
    for example in dataset:
        dropped_str = ' '.join(map(str, example['dropped_tokens']))
        ground_truth = example['original_tokens']
        # 将整数转换为字符串后再join
        gt_str = ' '.join(str(token) for token in ground_truth)
        gt_token_len = len(tokenizer(gt_str, add_special_tokens=False)["input_ids"])
        print(f"Ground truth token length: {gt_token_len}")
        ground_truths.append(ground_truth)
        chat = [
            {"role": "user", "content": f"{INSTRUCTION_PROMPT}\n{dropped_str}"}
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        print("predicting tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gt_token_len+10,
                do_sample=False,  # 贪婪解码
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[-1]
        predicted_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        predicted_tokens = list(map(int, predicted_text.strip().split()))
        print(f"ground_truth_length: {len(ground_truth)}")
        print(f"predicted_tokens_length: {len(predicted_tokens)}")
        if len(predicted_tokens) > len(ground_truth):
            predicted_tokens = predicted_tokens[:len(ground_truth)]

        predictions.append(predicted_tokens)
        total_tokens.extend(predicted_tokens)
        results.append({"predicted_tokens": predicted_tokens})

    with open(PREDICTIONS_PATH, 'w') as f:
        f.write('[\n')
        for item in results:
            f.write(json.dumps(item, separators=(',', ':')) + ',\n')
        f.write(']\n')
    print(f"Predictions saved to: {PREDICTIONS_PATH}")

    return total_tokens

def repair_audio(total_tokens):
    n_quantizers = 4
    total_length = len(total_tokens)
    t = total_length // n_quantizers
    codes = np.array(total_tokens).reshape(t, n_quantizers).transpose(1,0)
    codes = torch.tensor(codes, device=device).unsqueeze(0) 
    with torch.no_grad():
        latent,_,_ = encoder.quantizer.from_codes(codes)
        logamp_g, pha_g, _, _, y_g = decoder(latent)
        audio = y_g.squeeze()
        audio = audio.cpu().numpy()
        return audio


test_wav_dir = "/mnt/nvme_share/srt30/checkpoint/exp_new_llm"
print("Generating predictions...")
total_tokens = prediction(test_dataset)
audio_repaired = repair_audio(total_tokens)
sf.write(os.path.join(test_wav_dir, 'test_new.wav'), audio_repaired, h.sampling_rate,'PCM_16')