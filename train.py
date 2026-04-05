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

sys.path.append('src')
from model import HybridMathOCR
from tokenizer import LatexTokenizer
from dataset import MathOCRDataset, collate_fn
from augmentations import get_training_transforms

# --- CONFIG ---
BATCH_SIZE = 16 
ACCUM_STEPS = 4 
LR = 3e-4 # Slightly higher for Transformer breakthrough
EPOCHS = 50
DEVICE = torch.device('cuda')

# Essential for Blackwell Tensor Cores
torch.set_float32_matmul_precision('high')

def generate_causal_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(DEVICE)

def train():
    os.makedirs('checkpoints', exist_ok=True)
    writer = SummaryWriter('logs/ocr_fixed_run')

    # 1. Data Setup
    with open('data/processed/vocab.json', 'r') as f:
        vocab = json.load(f)
    tokenizer = LatexTokenizer()
    tokenizer.word2idx = vocab
    tokenizer.vocab_size = len(vocab)

    dataset = MathOCRDataset(
        json_path='data/processed/ground_truth.json',
        tokenizer=tokenizer,
        transforms=get_training_transforms()
    )
    
    train_size = int(0.95 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. Model & Compile
    model_inst = HybridMathOCR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    
    print("⌛ Compiling for Blackwell...")
    model_inst = torch.compile(model_inst)

    # 3. Optimization
    optimizer = optim.AdamW(model_inst.parameters(), lr=LR, weight_decay=1e-5)
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX, label_smoothing=0.1)
    scaler = GradScaler('cuda')

    print(f"🚀 Training {len(train_set)} images. Watch TensorBoard for the drop!")

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

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # preds is (Seq, Batch, Vocab)
                preds = model_inst(images, tgt_input, tgt_mask=tgt_mask)
                
                # DIMENSION FIX: Align to (Batch, Seq, Vocab)
                preds = preds.permute(1, 0, 2)
                
                loss = criterion(
                    preds.reshape(-1, tokenizer.vocab_size), 
                    tgt_expected.reshape(-1)
                )
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                if global_step % 10 == 0:
                    writer.add_scalar('Loss/Train_Step', loss.item() * ACCUM_STEPS, global_step)
                    writer.flush()
                global_step += 1

            pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)
            epoch_loss += loss.item() * ACCUM_STEPS

        # Validation
        model_inst.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                ti, te = tgts[:, :-1], tgts[:, 1:]
                tm = generate_causal_mask(ti.size(1))
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    p = model_inst(imgs, ti, tgt_mask=tm).permute(1, 0, 2)
                    l = criterion(p.reshape(-1, tokenizer.vocab_size), te.reshape(-1))
                v_loss += l.item()

        avg_v = v_loss / len(val_loader)
        writer.add_scalar('Loss/Val_Epoch', avg_v, epoch)
        print(f"✅ Epoch {epoch+1} | Train: {epoch_loss/len(train_loader):.4f} | Val: {avg_v:.4f}")
        
        torch.save(model_inst.state_dict(), f"checkpoints/ocr_v2_e{epoch+1}.pt")

if __name__ == "__main__":
    train()