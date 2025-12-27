import numpy as np 
import librosa 
import librosa.display
from matplotlib import pyplot as plt

import os
test_file = 'data/raw/1-19898-C-41.wav'

if not os.path.exists(test_file):
    import glob
    files = glob.glob('data/raw/*.wav')
    test_file = files[0]
    
print(f"Analysing : {os.path.basename(test_file)}")    

y,sr= librosa.load(test_file, sr=32000)   

M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, power=2, n_fft= 1024, n_mels=128)

M_db = librosa.power_to_db(M, ref= np.max) 
 
P = librosa.pcen(M*(2**31), eps= 1e-6, sr=sr, gain=0.98, bias=2, power= 0.5, time_constant= 0.400,hop_length=512)

plt.figure(figsize=(12, 10)) 

plt.suptitle(f"Using {os.path.basename(test_file)}")

plt.subplot(2,1,1)
img1 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')    
plt.colorbar(img1, format='%+2.0f dB')
plt.title('Mel-Spectrogram')


plt.subplot(2,1,2)
img2 = librosa.display.specshow(P,x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
plt.colorbar(img2)
plt.title('PCEN Spectrogram')

plt.tight_layout()      
plt.show()