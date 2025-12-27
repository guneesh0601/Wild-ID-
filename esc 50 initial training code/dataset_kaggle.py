import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
import os

class WildIDDataset(Dataset):
    def __init__(self, folds):
        
        self.annotations = pd.read_csv("/kaggle/input/environmental-sound-classification-50/esc50.csv")
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        self.annotations.reset_index(inplace=True,drop=True)
        
        self.root_dir = "/kaggle/input/environmental-sound-classification-50/audio/audio/44100/"
        self.sr = 32000
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 128

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):   
        
        audio_sample_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        
        label = self.annotations.iloc[index, 2]

        y,sr= librosa.load(audio_sample_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels,
            power=2.0 
        )

        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).view(1,128,313)
        
        """I have taken the dimensions to be 128x313 beacuse n_mels=128 so 128 rows in the spectrogram and 
           the  columns are 313 beacuse total columns = (total samples/hop length) :-
                                                      =  (5x32000)/512 = 312.5 nearly 313"""
        return mel_spec_tensor, label

