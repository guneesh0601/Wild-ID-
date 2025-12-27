
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from datasets import load_dataset
from dataset_xeno_stable import WildIDStreamDataset  # ← UPDATED
from model_xeno import WildIDClassifier
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # LOWER for stability
NUM_EPOCHS = 5
NUM_CLASSES = 10127
TRAIN_SAMPLES = 135000  # 135K train
VAL_SAMPLES = 15000     # 15K val = 150K TOTAL

def train():
    print(f" FULL 150K Xeno-Canto Training on: {DEVICE}")
    
    ds = load_dataset("ilyassmoummad/Xeno-Canto-6s-16khz", split="train", streaming=True)
    full_subset = ds.take(150000)
    train_raw = full_subset.take(TRAIN_SAMPLES)
    val_raw = full_subset.skip(TRAIN_SAMPLES).take(VAL_SAMPLES)
    
    model = WildIDClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    # Transfer learning (Conv/PCEN only)
    input_paths = ["/kaggle/input/wild-id-xeno-canto-training-script-1/best_model_wild_id.pth",
                   "/kaggle/working/best_model_wild_id.pth"]
    
    checkpoint = None
    for path in input_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            print(f"✅ ESC-50 weights: {path}")
            break

    if checkpoint:
        model_keys = set(model.state_dict().keys())
        backbone_weights = {k: v for k, v in checkpoint.items() 
                           if k in model_keys and "classifier" not in k and "lstm" not in k}
        model.load_state_dict(backbone_weights, strict=False)
        print("✅ Conv/PCEN loaded from ESC-50")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/5 - 150K SAMPLES")
        print(f"{'='*70}")
        
        for phase in ['train', 'val']:
            target_samples = TRAIN_SAMPLES if phase == 'train' else VAL_SAMPLES
            dataset_raw = train_raw if phase == 'train' else val_raw
            model.train() if phase == 'train' else model.eval()
            
            running_loss, correct_preds, total_samples, batch_count = 0.0, 0, 0, 0
            
            print(f"{phase.upper()}: Target {target_samples:,} samples")
            dataset_iter = iter(WildIDStreamDataset(dataset_raw))
            
            while total_samples < target_samples:
                try:
                    batch_inputs = []
                    batch_labels = []
                    
                    for _ in range(BATCH_SIZE):
                        mel_spec, label = next(dataset_iter)
                        batch_inputs.append(mel_spec)
                        batch_labels.append(label)
                    
                    inputs = torch.stack(batch_inputs).to(DEVICE)
                    labels = torch.stack(batch_labels).to(DEVICE)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # NaN CHECK
                        if torch.isnan(loss):
                            print("⚠️ NaN detected - skipping batch")
                            continue
                        
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                    
                    running_loss += loss.item() * BATCH_SIZE
                    _, predicted = torch.max(outputs, 1)
                    total_samples += BATCH_SIZE
                    correct_preds += (predicted == labels).sum().item()
                    batch_count += 1
                    
                    if batch_count % 100 == 0:
                        acc = correct_preds / total_samples
                        print(f"  B{batch_count:4d} | {total_samples:6d}/{target_samples:,} | "
                              f"Loss: {loss.item():.4f} | Acc: {acc:.4f}")
                
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Batch error: {e}")
                    continue
            
            epoch_loss = running_loss / total_samples if total_samples > 0 else 0
            epoch_acc = correct_preds / total_samples if total_samples > 0 else 0
            print(f"{phase.upper()}: Loss={epoch_loss:.4f} | Acc={epoch_acc:.4f} | "
                  f"Samples={total_samples:,}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "/kaggle/working/best_model_xeno_150k.pth")
                print(f" BEST MODEL! Val Acc: {best_acc:.4f}")
    
    print(f"\n🏆 150K TRAINING COMPLETE! Best Acc: {best_acc:.4f}")
    print("Model saved: /kaggle/working/best_model_xeno_150k.pth")

if __name__ == "__main__":
    train()
