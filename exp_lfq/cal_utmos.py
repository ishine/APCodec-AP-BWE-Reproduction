import torch
import torchaudio
import librosa
import os
import pandas as pd

# 输入输出路径
input_dir = "/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_LFQ/output_wav_1140k"
output_csv = "/mnt/nvme_share/srt30/APCodec-AP-BWE-Reproduction/exp_LFQ/file.csv"

# 加载 SpeechMOS 模型
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

# 存储结果
results = []

# 遍历音频文件
for file in os.listdir(input_dir):
    if file.endswith(".wav"):
        # 加载音频
        audio_path = os.path.join(input_dir, file)
        wave, sr = librosa.load(audio_path, sr=48000, mono=True)
        
        # 转换为张量
        wave_tensor = torch.from_numpy(wave).unsqueeze(0)  # Shape: (1, samples)
        
        # 预测 MOS 分数
        score = predictor(wave_tensor, sr)
        
        # 保存结果
        results.append({"filename": file, "mos_score": score.item()})
        

# 保存到 CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")

# 计算平均 MOS
avg_mos = df["mos_score"].mean()
print(f"Average MOS Score: {avg_mos:.4f}")