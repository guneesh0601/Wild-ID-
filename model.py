import torch
import torch.nn as nn
import torchaudio


# PCEN layer (first layer of my Neural Network)
class TrainablePCEN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_delta = nn.Parameter(torch.tensor(0.0))
        self.log_r = nn.Parameter(torch.tensor(0.0))
        
        self.eps = 1e-6
        self.s = 0.025

    def forward(self, x):      # x is input 4D tensor for pytorch, it is of size [batchsize,channels,height,width]
        
        alpha = self.log_alpha.exp() * 0.98  
        delta = self.log_delta.exp() * 2.0   
        r     = self.log_r.exp() * 0.5       

    
        x_squeezed = x.squeeze(1)
        
        # PCEN MATH
        smoothed_energy = []
        last_m = torch.zeros_like(x_squeezed[:, :, 0])
        s = 0.025 
        
        for t in range(x_squeezed.size(-1)):
            last_m = (1 - s) * last_m + s * x_squeezed[:, :, t]
            smoothed_energy.append(last_m.unsqueeze(-1))
            
        M = torch.cat(smoothed_energy, dim=-1)
        pcen_output = (x_squeezed / (self.eps + M).pow(alpha) + delta).pow(r) - delta.pow(r)

    
        return pcen_output.unsqueeze(1)
    
    
    
# CNN layer  
class WildIDClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        
        self.front_end = TrainablePCEN()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2), 
            nn.BatchNorm2d(64),                         
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)                             
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)
        )
        
        # Block 4 
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)
        )
        
        # LSTM layerrs
        self.lstm = nn.LSTM(
            input_size=128 * 8, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Classifier layer FC layer
        self.classifier = nn.Linear(128 * 2, num_classes)
        
        
        

    def forward(self, x):
        
        x = self.front_end(x)
        
        x = self.block1(x) 
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.block4(x) 
        
        x = x.permute(0, 3, 1, 2) 
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3)) 
       
        
        x, _ = self.lstm(x)
        
        x = x.mean(dim=1) 
        
        x = self.classifier(x)
        
        return x