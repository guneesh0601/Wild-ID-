import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import os

from dataset import WildIDDataset
from model import WildIDClassifier

DEVICE = torch.device("cuda")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

TRAIN_FOLDS = [1, 2, 3, 4]
VAL_FOLDS = [5]

def train():
    print(f"Starting Wild-ID Training on: {DEVICE}")
    
    train_dataset = WildIDDataset(folds=TRAIN_FOLDS)
    val_dataset = WildIDDataset(folds=VAL_FOLDS)

    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )

    model = WildIDClassifier(num_classes=50).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    spec_augmenter = nn.Sequential(
        T.FrequencyMasking(freq_mask_param=15),
        T.TimeMasking(time_mask_param=35)
    ).to(DEVICE)

    # Nowww train
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                loader = train_loader
            else:
                model.eval()   
                loader = val_loader

            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                
                if phase == 'train':
                    inputs = spec_augmenter(inputs)

                optimizer.zero_grad()

                # Forwardd pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backprop (&optimize)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

            epoch_loss = running_loss / total_preds
            epoch_acc = correct_preds / total_preds
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # save the best model paarameters
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model_wild_id.pth")
                print(f"--> New Best Model Saved! (Acc: {best_acc:.4f})")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()