import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from dataset_newmodel import WildIDStreamDataset 
from model_newmodel import WildIDClassifier
import os
import zlib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
NUM_CLASSES = 10127

NUM_EPOCHS = 10
TRAIN_LIMIT = 250000
VAL_LIMIT = 5000

def get_split_label(example):
    key = example.get('__key__', str(example))
    hash_val = zlib.adler32(key.encode())
    mod_val = hash_val % 100
    
    if mod_val < 10: 
        return 'test'
    elif mod_val < 20:
        return 'val'
    return 'train'

def train():
    print(f" STARTING training ON {DEVICE}")
    
    ds = load_dataset("ilyassmoummad/Xeno-Canto-6s-16khz", split="train", streaming=True)
    
    ds = ds.shuffle(seed=42, buffer_size=10000)
    
    val_stream = ds.filter(lambda x: get_split_label(x) == 'val')
    train_stream = ds.filter(lambda x: get_split_label(x) == 'train')
    
    model = WildIDClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    
    checkpoint_path = "last_model_xeno.pth"
    
    if os.path.exists(checkpoint_path):
        print(f" Found {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(" Weights loaded successfully!")
        except Exception as e:
            print(f" Error loading weights: {e}")
            
    else:
        if os.path.exists("best_model_xeno_4.pth"):
             print(f" Found best_model_xeno_4.pth. Resuming...")
             model.load_state_dict(torch.load("best_model_xeno_4.pth", map_location=DEVICE))
        else:
             print(" No saved weights")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch+1}/{NUM_EPOCHS} (Session Epoch)\n{'='*60}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                dataset_iter = iter(WildIDStreamDataset(
                    train_stream, 
                    num_classes=NUM_CLASSES, 
                    augment=True
                ))
                target_samples = TRAIN_LIMIT
                model.train()
            else:
                dataset_iter = iter(WildIDStreamDataset(
                    val_stream, 
                    num_classes=NUM_CLASSES, 
                    augment=False
                ))
                target_samples = VAL_LIMIT
                model.eval()
            
            running_loss, correct_preds, total_samples, batch_count = 0.0, 0, 0, 0
            
            while total_samples < target_samples:
                batch_inputs, batch_labels = [], []
                 
                while len(batch_inputs) < BATCH_SIZE:
                    try:
                        mel_spec, label = next(dataset_iter)
                        batch_inputs.append(mel_spec)
                        batch_labels.append(label)
                    except StopIteration: 
                        break 
                    except Exception: 
                        continue 
                
                if not batch_inputs: 
                    break
                
                inputs = torch.stack(batch_inputs).to(DEVICE)
                labels = torch.stack(batch_labels).to(DEVICE)
                
                if phase == 'train': 
                    optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                
                bs = inputs.size(0)
                running_loss += loss.item() * bs
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_samples += bs
                batch_count += 1
                
                if batch_count % 100 == 0:
                    acc = correct_preds / total_samples
                    print(f" {phase.upper()} | B{batch_count} | Samples {total_samples}/{target_samples} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

            epoch_acc = correct_preds / total_samples if total_samples > 0 else 0
            print(f" {phase.upper()} END: Acc={epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model_xeno_5.pth")
                print(f" NEW BEST MODEL SAVED!")

    print(" Saving 'last_model_xeno.pth' ")
    torch.save(model.state_dict(), "last_model_xeno.pth")
    

if __name__ == "__main__":
    train()