import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms(image_size=(256, 256)):
    return A.Compose([
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.Morphological(operation='erosion', scale=(2, 2), p=0.5), 
            A.Morphological(operation='dilation', scale=(2, 2), p=0.5),
        ], p=0.2),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])