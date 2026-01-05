import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
# ⚠️ MAKE SURE YOU HAVE THESE 2 FILES IN YOUR FOLDER
from dataset_xeno import WildIDStreamDataset 
from model_xeno import WildIDClassifier
import os
import zlib

# --- CONFIGURATION ---
DEVICE = torch.device("cuda")
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
NUM_CLASSES = 10127

# --- YOUR STRATEGY SETTINGS ---
NUM_EPOCHS = 10          # Do 3 epochs per session
TRAIN_LIMIT = 250000     # The "Golden" 250k subset
VAL_LIMIT = 5000         # Check accuracy on 5k files

def get_split_label(example):
    """
    Deterministically splits data based on the filename hash.
    0-10%   -> Test
    10-20%  -> Validation
    20-100% -> Train
    """
    key = example.get('__key__', str(example))
    hash_val = zlib.adler32(key.encode())
    mod_val = hash_val % 100
    
    if mod_val < 10: 
        return 'test'
    elif mod_val < 20:
        return 'val'
    return 'train'

def train():
    print(f"🚀 STARTING STRATEGY: 250k FIXED SUBSET ON {DEVICE}")
    
    # 1. Load Streaming Dataset
    ds = load_dataset("ilyassmoummad/Xeno-Canto-6s-16khz", split="train", streaming=True)
    
    # 🔥 CRITICAL FIX: Shuffle with a fixed seed.
    # This ensures we get a random mix of birds, but the SAME mix every time we run.
    ds = ds.shuffle(seed=42, buffer_size=10000)
    
    # Apply the split filters
    # Note: Because of the fixed seed above, 'train_stream' will deliver the
    # exact same files in the exact same order every time you run this script.
    val_stream = ds.filter(lambda x: get_split_label(x) == 'val')
    train_stream = ds.filter(lambda x: get_split_label(x) == 'train')
    
    model = WildIDClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 2. Smart Resume Logic
    # We look for 'last_model' first (to continue training), then 'best_model'.
    start_epoch = 0
    # Note: Using the specific path from your screenshot
    if os.path.exists("/kaggle/input/best-model-4-normal-model/best_model_xeno_4.pth"):
        print(f"📥 Using Transfer learning . Resuming...")
        model.load_state_dict(torch.load("/kaggle/input/best-model-4-normal-model/best_model_xeno_4.pth"))
    else:
        print("✨ No saved weights found. Starting fresh.")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch+1}/{NUM_EPOCHS} (Session Epoch)\n{'='*60}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                # Re-create iterator to start from the top of the 250k list
                dataset_iter = iter(WildIDStreamDataset(train_stream, num_classes=NUM_CLASSES, augment=True))
                target_samples = TRAIN_LIMIT
                model.train()
            else:
                dataset_iter = iter(WildIDStreamDataset(val_stream, num_classes=NUM_CLASSES, augment=False))
                target_samples = VAL_LIMIT
                model.eval()
            
            running_loss, correct_preds, total_samples, batch_count = 0.0, 0, 0, 0
            
            # Loop until we hit our limit (250k for train, 5k for val)
            while total_samples < target_samples:
                batch_inputs, batch_labels = [], []
                 
                # Fetch Batch
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
                
                # Zero Gradients
                if phase == 'train': 
                    optimizer.zero_grad()
                
                # Forward Pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                
                # Stats
                bs = inputs.size(0)
                running_loss += loss.item() * bs
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_samples += bs
                batch_count += 1
                
                if batch_count % 100 == 0:
                    acc = correct_preds / total_samples
                    print(f" {phase.upper()} | B{batch_count} | Samples {total_samples}/{target_samples} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

            # End of Phase Stats
            epoch_acc = correct_preds / total_samples if total_samples > 0 else 0
            print(f"✅ {phase.upper()} END: Acc={epoch_acc:.4f}")
            
            # Save Checkpoints
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model_xeno_5.pth")
                print(f"🏆 NEW BEST MODEL SAVED!")

    # 4. Save Final State for Next Session
    print("💾 Saving 'last_model_xeno.pth' for next session...")
    torch.save(model.state_dict(), "last_model_xeno.pth")
    print("--- Session Complete. Re-run script to do 3 more epochs. ---")

if __name__ == "__main__":
    train()