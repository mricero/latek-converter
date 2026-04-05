import os
import sys
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# Metric Library
try:
    from Levenshtein import distance as lev_dist
except ImportError:
    print("❌ Error: Please run 'pip install python-Levenshtein' first.")
    sys.exit()

# Add src to path for local imports
sys.path.append('src')
from model import HybridMathOCR
from tokenizer import LatexTokenizer
from dataset import MathOCRDataset, collate_fn

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "checkpoints/ocr_v2_e27.pt" # Update to your best file
MAX_LEN = 150
BATCH_SIZE = 1 # Keep at 1 for clean greedy decoding

def test():
    # 1. Setup Tokenizer
    vocab_path = 'data/processed/vocab.json'
    if not os.path.exists(vocab_path):
        print(f"❌ Error: {vocab_path} not found.")
        return

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    tokenizer = LatexTokenizer()
    tokenizer.word2idx = vocab
    tokenizer.vocab_size = len(vocab)
    tokenizer.idx2word = {v: k for k, v in vocab.items()}

    # 2. Setup Dataset (Accessing the Test/Val Split)
    dataset = MathOCRDataset(
        json_path='data/processed/ground_truth.json',
        tokenizer=tokenizer,
        transforms=None # NEVER use augmentations during testing
    )
    
    # We use a fixed seed (42) to ensure we test on the SAME images 
    # that the model saw during the validation phase of training.
    train_size = int(0.95 * len(dataset))
    _, test_set = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. Model Initialization & Smart Load
    model = HybridMathOCR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    
    print(f"🔄 Loading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Error: Checkpoint file not found.")
        return

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # --- SMART LOAD LOGIC ---
    # Handles both direct state_dicts and wrapped dictionaries with metadata
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print(f"✅ Detected Metadata Wrapper (Epoch: {ckpt.get('epoch', 'N/A')})")
    else:
        state_dict = ckpt
        print("✅ Detected Direct Weights File")

    # Strip the '_orig_mod.' prefix added by torch.compile
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 4. Evaluation Loop
    total_cer = 0
    total_chars = 0
    perfect_matches = 0
    
    print(f"🧪 Evaluating {len(test_set)} images on {DEVICE}...")

    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            images = images.to(DEVICE)
            
            # Start sequence with <SOS>
            generated = [tokenizer.word2idx['<SOS>']]
            
            # Greedy Decoding Loop
            for _ in range(MAX_LEN):
                tgt_tensor = torch.LongTensor(generated).unsqueeze(0).to(DEVICE)
                
                # Forward Pass
                output = model(images, tgt_tensor) 
                
                # Get the last predicted token (output is [Seq, Batch, Vocab])
                # We want the last step of the sequence, the only batch, and the max logit
                logits = output[-1, 0, :] 
                next_token = torch.argmax(logits).item()
                
                generated.append(next_token)
                
                # Stop if <EOS> is reached
                if next_token == tokenizer.word2idx['<EOS>']:
                    break
            
            # 5. Convert Indices to LaTeX Strings
            # Slice [1:-1] to remove <SOS> and <EOS> tags
            pred_str = tokenizer.decode(generated).replace('<SOS>', '').replace('<EOS>', '').strip()
            target_str = tokenizer.decode(targets[0].tolist()).replace('<SOS>', '').replace('<EOS>', '').strip()

            # 6. Calculate Metrics
            dist = lev_dist(pred_str, target_str)
            total_cer += dist
            total_chars += max(len(target_str), 1)
            
            if pred_str == target_str:
                perfect_matches += 1

            # Print first few samples to see the progress
            if i < 3:
                print(f"\n--- Sample {i+1} ---")
                print(f"Target: {target_str}")
                print(f"Pred  : {pred_str}")
                print(f"Dist  : {dist} chars")

    # 7. Final Statistics
    avg_cer = (total_cer / total_chars) * 100
    exp_rate = (perfect_matches / len(test_set)) * 100

    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Character Error Rate (CER): {avg_cer:.2f}%")
    print(f"Expression Rate (ExpRate):  {exp_rate:.2f}%")
    print(f"Perfect Matches: {perfect_matches} / {len(test_set)}")
    print("="*30)

if __name__ == "__main__":
    test()
