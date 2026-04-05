import os
import json
from datasets import load_dataset
from tqdm import tqdm
import sys

sys.path.append('src')
from tokenizer import LatexTokenizer

def process_iam_parquets():
    print("\n--- 1. Extracting IAM Images from Parquet ---")
    iam_img_dir = "data/raw/IAM/images_extracted"
    os.makedirs(iam_img_dir, exist_ok=True)
    
    master_dict = {}
    try:
        ds = load_dataset("parquet", data_files={"train": "data/raw/IAM/data/train*.parquet"})['train']
        for i, item in enumerate(tqdm(ds, desc="Saving IAM PNGs")):
            img = item['image']
            text = item['text']
            
            img_filename = f"iam_{i}.png"
            img_path = os.path.join(iam_img_dir, img_filename)
            img.save(img_path)
            
            spaced_text = " ".join(list(text))
            master_dict[f"data/raw/IAM/images_extracted/{img_filename}"] = spaced_text
            
        print(f"✅ Extracted {len(master_dict)} IAM images.")
    except Exception as e:
        print(f"❌ Failed to extract IAM: {e}")
        
    return master_dict

def process_hme100k():
    print("\n--- 2. Parsing HME100K Labels ---")
    master_dict = {}
    
    # Hardcoded exactly to your VS Code directory structure
    label_path = "data/raw/HME100K/HME100K/train_images/train_labels.txt"
    img_dir = "data/raw/HME100K/HME100K/train_images"
    
    if not os.path.exists(label_path):
        print(f"❌ Cannot find {label_path}. Please check spelling.")
        return master_dict
        
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_name = parts[0] # e.g., train_0.jpg
                    formula = parts[1]
                    
                    # Combine folder and filename
                    rel_img_path = f"{img_dir}/{img_name}"
                    master_dict[rel_img_path] = formula
        print(f"✅ Parsed {len(master_dict)} HME100K labels.")
    except Exception as e:
        print(f"❌ Failed to parse HME100K: {e}")
        
    return master_dict

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Gather all data
    iam_dict = process_iam_parquets()
    hme_dict = process_hme100k()
    
    # 2. Merge into one massive dictionary
    final_dataset = {**iam_dict, **hme_dict}
    print(f"\n🚀 Total Unified Images ready for PyTorch: {len(final_dataset)}")
    
    # 3. Save the Ground Truth JSON
    gt_path = "data/processed/ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(final_dataset, f, indent=4)
    print(f"💾 Saved master label map to {gt_path}")
    
    # 4. Build the Frequency-Filtered Vocabulary
    print("\n--- 3. Building Tokenizer Vocabulary ---")
    tokenizer = LatexTokenizer(min_freq=3)
    tokenizer.fit_on_texts(list(final_dataset.values()))
    
    vocab_path = "data/processed/vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(tokenizer.word2idx, f, indent=4)
    print(f"💾 Saved Vocabulary ({tokenizer.vocab_size} tokens) to {vocab_path}")
    
    print("\n🏆 Phase 1 is 100% Complete.")