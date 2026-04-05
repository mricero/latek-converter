import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from augmentations import get_base_transforms, get_stochastic_augs
import albumentations as A

class MathOCRDataset(Dataset):
    def __init__(self, json_path, tokenizer, is_train=True, dist_type='bell'):
        with open(json_path, 'r') as f:
            import json
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.dist_type = dist_type # 'bell' or 'uniform'
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        formula = self.data[img_name]
        
        # 1. Load Image
        image = cv2.imread(f"data/raw/{img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Sample Augmentation Intensity
        if self.is_train:
            # Beta(4,4) creates a beautiful bell curve centered at 0.5
            # Beta(1,1) would be Uniform [0,1]
            alpha, beta = (4, 4) if self.dist_type == 'bell' else (1, 1)
            intensity = np.random.beta(alpha, beta)
            
            aug_list = get_stochastic_augs(intensity) + get_base_transforms()
        else:
            intensity = 0 # No noise for Val/Test
            aug_list = get_base_transforms()

        transform = A.Compose(aug_list)
        image_tensor = transform(image=image)['image']

        # 3. Tokenize
        tokens = self.tokenizer.encode(formula)
        return image_tensor, torch.tensor(tokens)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    # Pad sequences to match the longest in the batch
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return images, targets