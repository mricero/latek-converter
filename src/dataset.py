import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class MathOCRDataset(Dataset):
    def __init__(self, json_path, tokenizer, transforms=None):
        """
        json_path: Path to 'data/processed/ground_truth.json'
        tokenizer: Your LatexTokenizer instance
        transforms: Albumentations warping pipeline
        """
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)
        
        self.image_paths = list(self.data_dict.keys())
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Get Path and Formula
        img_path = self.image_paths[idx]
        formula = self.data_dict[img_path]

        # 2. Load Image
        # Since your JSON keys are already 'data/raw/...', we load directly
        try:
            image = Image.open(img_path).convert('L')
            image_np = np.array(image)
        except Exception as e:
            # If an image is missing, return a blank 256x256 image
            image_np = np.zeros((256, 256), dtype=np.uint8)

        # 3. Apply Warping (Albumentations)
        if self.transforms:
            augmented = self.transforms(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0) / 255.0

        # 4. Tokenize
        tokens = self.tokenizer.encode(formula)

        return image_tensor, tokens

def collate_fn(batch):
    """
    Groups images into a batch and pads sequences to the same length.
    """
    images, targets = zip(*batch)
    
    # Stack images: (B, 1, H, W)
    images = torch.stack(images, dim=0)
    
    # Pad sequences: (B, Max_Seq_Len_In_Batch)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return images, targets_padded