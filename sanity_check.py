import torch
import sys
import os

# Add src to path
sys.path.append('src')
from model import HybridMathOCR
from tokenizer import LatexTokenizer

def run_sanity_check():
    print("🚀 Starting Sanity Check...")
    
    # 1. Setup Mock Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = 200 # Roughly what your dataset has
    model = HybridMathOCR(vocab_size=vocab_size).to(device)
    model.eval()
    
    # Mock Batch: 2 images of 256x256, 2 target sequences of length 20
    dummy_images = torch.randn(2, 1, 256, 256).to(device)
    dummy_targets = torch.randint(0, vocab_size, (2, 20)).to(device)
    
    # 2. Test Forward Pass
    print("Testing forward pass...")
    try:
        # Transformers expect targets to be shifted for training
        # But for a simple shape check, we just pass them through
        with torch.no_grad():
            output = model(dummy_images, dummy_targets)
        
        print(f"✅ Success! Output shape: {output.shape}") 
        # Expected: [Sequence_Len, Batch, Vocab_Size] -> [20, 2, 200]
        
        # 3. Parameter Count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total Parameters: {total_params / 1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ Sanity Check Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_sanity_check()