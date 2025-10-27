import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "/mnt/nvme_share/srt30/checkpoint/exp_500/output_wav_540k/p360_001.wav"

y, sr = librosa.load(audio_path, sr=8000)

D = librosa.amplitude_to_db(abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)

# 画频谱图
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, hop_length=256, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.tight_layout()
plt.savefig("/mnt/nvme_share/srt30/APCodec-Reproduction/exp_500/spectrogram.png")   # 保存到当前目录
print("频谱图已保存为 spectrogram.png")