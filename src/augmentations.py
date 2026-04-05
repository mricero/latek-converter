import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms(image_size=(256, 256)):
    """
    Updated for Albumentations 2.0+ API.
    Uses std_range for noise and Morphological for ink effects.
    """
    return A.Compose([
        # 1. Spatial Distortions
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        
        # 2. Noise (Updated from var_limit to std_range)
        A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        
        # 3. Ink Bleed or Fading (Using the unified Morphological transform)
        A.OneOf([
            A.Morphological(operation='erosion', scale=(2, 2), p=0.5), 
            A.Morphological(operation='dilation', scale=(2, 2), p=0.5),
        ], p=0.3),
        
        # 4. Standardize Size and Format
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])