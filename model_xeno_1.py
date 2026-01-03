import torch
import torch.nn as nn

class TrainablePCEN(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))
        self.log_delta = nn.Parameter(torch.tensor(0.0))
        self.log_r = nn.Parameter(torch.tensor(-1.0))
        self.eps = 1e-6

    def forward(self, x):
        
        alpha = torch.clamp(self.log_alpha.exp(), 0.01, 1.0)
        delta = torch.clamp(self.log_delta.exp(), 0.01, 5.0)
        r = torch.clamp(self.log_r.exp(), 0.01, 1.0)
        
        
        x = torch.exp(x) 
        x_squeezed = x.squeeze(1)
        
        # PCEN Energy Smoothing
        smoothed_energy = []
        last_m = torch.zeros_like(x_squeezed[:, :, 0])
        s = 0.025
        
        for t in range(x_squeezed.size(-1)):
            last_m = (1 - s) * last_m + s * x_squeezed[:, :, t]
            smoothed_energy.append(last_m.unsqueeze(-1))
            
        M = torch.cat(smoothed_energy, dim=-1)
        
        #PCEN formula
        pcen_output = (x_squeezed / (self.eps + M).pow(alpha) + delta).pow(r) - delta.pow(r)
        
        return pcen_output.unsqueeze(1)

class WildIDClassifier(nn.Module):
    def __init__(self, num_classes=10127): 
        super().__init__()
        # 1. PCEN layer
        self.front_end = TrainablePCEN()
        
        #  CNN Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2), 
            nn.BatchNorm2d(64),                         
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)                             
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),           
            nn.Dropout(0.2)
        )
        
        # LSTM layers 
        self.lstm = nn.LSTM(
            input_size=128 * 8,  
            hidden_size=128, 
            num_layers=4,             
            batch_first=True, 
            bidirectional=True
        )
        

        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(128 * 2, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 4 (Output)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        
        x = self.front_end(x)
        
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Reshape for LSTM
        x = x.permute(0, 3, 1, 2) 
        x = x.reshape(x.size(0), x.size(1), (x.size(2)*x.size(3))) 
        
        
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        
        return self.classifier(x)