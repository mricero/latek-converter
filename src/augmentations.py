import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_base_pipeline():
    """Essential transforms that always happen."""
    return [
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]

def get_heavy_augs(intensity=1.0):
    """
    Returns a list of transforms where 'intensity' scales 
    either the probability or the magnitude.
    """
    return [
        A.Perspective(scale=(0.02 * intensity, 0.08 * intensity), p=0.5 * intensity),
        A.ElasticTransform(alpha=1 * intensity, sigma=50, p=0.4 * intensity),
        A.CoarseDropout(max_holes=int(6 * intensity), max_height=16, max_width=16, p=0.3 * intensity),
        A.GaussNoise(std_range=(0.1 * intensity, 0.3 * intensity), p=0.3 * intensity)
    ]