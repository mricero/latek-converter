import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import HybridMathOCR
from src.tokenizer import LatexTokenizer
from src.dataset import MathOCRDataset, collate_fn

# --- HYPERPARAMETERS ---
BATCH_SIZE = 16
ACCUM_STEPS = 4
LR = 3e-4
EPOCHS = 50
WEIGHT_DECAY = 0.01  # Strong L2 to prevent overfitting
MAX_GRAD_NORM = 1.0  # Clipping for Transformer stability
DEVICE = torch.device('cuda')

torch.set_float32_matmul_precision('high')

def generate_causal_mask(sz):
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(DEVICE)

def train():
    os.makedirs('checkpoints', exist_ok=True)
    writer = SummaryWriter('logs/ocr_stochastic_run')

    # 1. Load Tokenizer
    tokenizer = LatexTokenizer() 
    # (Assuming tokenizer.load_vocab is implemented or vocab is in src/)

    # 2. Data Splits
    dataset = MathOCRDataset('data/processed/ground_truth.json', tokenizer, is_train=True, dist_type='bell')
    train_size = int(0.95 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    
    # Crucial: Validation must be "Perfect Data" (Intensity 0)
    val_set.dataset.is_train = False 

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 3. Model & Optimization
    model = HybridMathOCR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model = torch.compile(model) # Blackwell acceleration

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for i, (imgs, tgts) in enumerate(pbar):
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            ti, te = tgts[:, :-1], tgts[:, 1:]
            tm = generate_causal_mask(ti.size(1))

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # .permute(1, 0, 2) FIX applied inside training
                preds = model(imgs, ti, tgt_mask=tm).permute(1, 0, 2)
                loss = criterion(preds.reshape(-1, tokenizer.vocab_size), te.reshape(-1)) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)

        # 4. Validation & Save Best
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                ti, te = tgts[:, :-1], tgts[:, 1:]
                tm = generate_causal_mask(ti.size(1))
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    p = model(imgs, ti, tgt_mask=tm).permute(1, 0, 2)
                    v_loss += criterion(p.reshape(-1, tokenizer.vocab_size), te.reshape(-1)).item()
        
        avg_v = v_loss / len(val_loader)
        writer.add_scalar('Loss/Val', avg_v, epoch)
        
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"⭐ New Best: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()