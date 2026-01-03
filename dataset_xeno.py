import torch
import torchaudio.transforms as T
import io
import zlib
import numpy as np

class WildIDStreamDataset:
    def __init__(self, hf_dataset, num_classes=10127):
        self.hf_dataset = hf_dataset
        self.num_classes = num_classes
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000, n_mels=128, n_fft=2048, hop_length=256
        )
        self.freq_mask = T.FrequencyMasking(freq_mask_param=27)
        self.time_mask = T.TimeMasking(time_mask_param=70)

    def _get_label_id(self, key_str):
        try:
            parts = key_str.split('/')
            species_code = parts[1] if len(parts) >= 2 else "unknown"
            return zlib.adler32(species_code.encode()) % self.num_classes
        except:
            return 0

    def __iter__(self):
        iterator = iter(self.hf_dataset)
        
        while True:
            try:
                item = next(iterator)
                
                # 1. LOAD FROM BLOB
                if 'pt' in item:
                    buffer = io.BytesIO(item['pt'])
                    audio = torch.load(buffer) 
                else:
                    continue

                # --- NUCLEAR SANITIZATION (The Fix) ---
                # 1. Replace NaNs/Infs with 0
                audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 2. Hard Clip huge audio values
                audio = torch.clamp(audio, -1.0, 1.0)
                # --------------------------------------

                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                
                # Length Check (6 seconds)
                if audio.size(1) > 96000:
                    audio = audio[:, :96000]
                elif audio.size(1) < 96000:
                    padding = 96000 - audio.size(1)
                    audio = torch.nn.functional.pad(audio, (0, padding))

                # Transform
                mel_spec = self.mel_transform(audio)
                
                # Safe Log
                mel_spec = torch.log(mel_spec + 1e-6)
                
                # Standardize
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
                
                # Final Clamp check
                mel_spec = torch.clamp(mel_spec, -10.0, 10.0)

                # Augment
                if torch.rand(1) < 0.5:
                    mel_spec = self.freq_mask(mel_spec)
                    mel_spec = self.time_mask(mel_spec)

                label_id = self._get_label_id(item.get('__key__', 'unknown/unknown'))
                yield mel_spec, torch.tensor(label_id).long()

            except StopIteration:
                break
            except Exception:
                continue