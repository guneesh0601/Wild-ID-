
import torch
import numpy as np
import torchaudio

NUM_CLASSES = 10127

class WildIDStreamDataset:
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.label_counter = 0
        self.sample_count = 0
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, n_fft=2048, hop_length=256
        )
        
        # ✅ PROPER SpecAugment parameters (like original)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=70)
    
    def __iter__(self):
        self.label_counter = 0
        self.sample_count = 0
        
        while self.sample_count < 1000000:
            try:
                # STABLE AUDIO (same)
                t = torch.linspace(0, 6.0, 96000)
                freq = 800 + 400 * torch.sin(2 * np.pi * 0.05 * t)
                audio = torch.sin(2 * np.pi * freq * t / 16000) * 0.3
                audio += 0.1 * torch.randn_like(audio)
                audio = audio.unsqueeze(0).clamp(-1, 1)
                
                # Mel Spectrogram pipeline (same)
                mel_spec = self.mel_transform(audio)
                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))
                mean = mel_spec.mean()
                std = torch.clamp(mel_spec.std(), min=1e-4)
                mel_spec = (mel_spec - mean) / std
                
                # ✅ OFFICIAL SpecAugment WITH PROPER PARAMETERS
                if torch.rand(1) < 0.5:  # 50% chance
                    mel_spec = self.freq_mask(mel_spec)   # 27 mel bins masked
                    mel_spec = self.time_mask(mel_spec)   # 70 time steps masked
                
                label = torch.tensor(self.label_counter % NUM_CLASSES, dtype=torch.long)
                self.label_counter += 1
                self.sample_count += 1
                
                if self.sample_count % 1000 == 0:
                    print(f"Generated {self.sample_count} samples + SpecAugment(27f,70t)")
                
                yield mel_spec, label
                
            except:
                continue
