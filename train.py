import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure local src directory is visible
sys.path.append('src')
from model import HybridMathOCR
from tokenizer import LatexTokenizer
from dataset import MathOCRDataset, collate_fn
from augmentations import get_training_transforms

# --- CONFIGURATION ---
BATCH_SIZE = 16 
ACCUM_STEPS = 4 
LR = 1e-4
EPOCHS = 50
DEVICE = torch.device('cuda')

# Blackwell Optimization: Essential for RTX 50-series Tensor Cores
torch.set_float32_matmul_precision('high')

def generate_causal_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(DEVICE)

def train():
    os.makedirs('checkpoints', exist_ok=True)
    # This creates the 'logs' folder where your graph data lives
    writer = SummaryWriter('logs/blackwell_ocr_v1')

    # 1. Load Data & Tokenizer
    with open('data/processed/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    tokenizer = LatexTokenizer()
    tokenizer.word2idx = vocab
    tokenizer.vocab_size = len(vocab)

    full_dataset = MathOCRDataset(
        json_path='data/processed/ground_truth.json',
        tokenizer=tokenizer,
        transforms=get_training_transforms()
    )
    
    train_size = int(0.95 * len(full_dataset))
    train_set, val_set = random_split(full_dataset, [train_size, len(full_dataset)-train_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. Model Initialization
    model_inst = HybridMathOCR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    
    print("⌛ Compiling 100M parameter model for Blackwell...")
    try:
        # Default compile is safest for Gradient Accumulation
        model_inst = torch.compile(model_inst)
        print("⚡ Model compiled successfully.")
    except Exception as e:
        print(f"⚠️ Compilation skipped: {e}")

    # 3. Optimization
    optimizer = optim.AdamW(model_inst.parameters(), lr=LR, weight_decay=1e-5)
    
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX, label_smoothing=0.1)
    scaler = GradScaler('cuda')

    print(f"🚀 Launching training on {len(train_set)} images...")

    global_step = 0
    for epoch in range(EPOCHS):
        model_inst.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (images, targets) in enumerate(pbar):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            tgt_input, tgt_expected = targets[:, :-1], targets[:, 1:]
            tgt_mask = generate_causal_mask(tgt_input.size(1))

            # BFloat16 is highly recommended for Blackwell stability
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model_inst(images, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(preds.reshape(-1, tokenizer.vocab_size), tgt_expected.reshape(-1))
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                # --- REAL-TIME LOGGING ---
                # Log the loss to TensorBoard every 10 optimized steps
                if global_step % 10 == 0:
                    writer.add_scalar('Loss/Train_Step', loss.item() * ACCUM_STEPS, global_step)
                    writer.add_scalar('LR/Step', scheduler.get_last_lr()[0], global_step)
                    writer.flush() # Force write to disk immediately
                
                global_step += 1

            epoch_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)

        # 4. Validation at end of Epoch
        model_inst.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                ti, te = tgts[:, :-1], tgts[:, 1:]
                tm = generate_causal_mask(ti.size(1))
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    p = model_inst(imgs, ti, tgt_mask=tm)
                    l = criterion(p.reshape(-1, tokenizer.vocab_size), te.reshape(-1))
                v_loss += l.item()

        avg_v = v_loss / len(val_loader)
        writer.add_scalar('Loss/Validation_Epoch', avg_v, epoch)
        print(f"✅ Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {avg_v:.4f}")
        
        torch.save({
            'model_state_dict': model_inst.state_dict(),
            'vocab': tokenizer.word2idx,
        }, f"checkpoints/ocr_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()