## 🏗️ Directory Architecture

The project is structured to separate data processing, model logic, and training execution.

```text
LATEK-CONVERTER/
├── src/                        # Core Logic
│   ├── model.py                # 101M Hybrid Architecture (ResNet50 + Transformer)
│   ├── tokenizer.py            # LatexTokenizer (Vocab management & Special Tokens)
│   ├── dataset.py              # MathOCRDataset & Collate logic
│   └── augmentations.py        # Albumentations pipeline (Elastic, Grid, Morphological)
├── data/                       # [LOCAL ONLY]
│   ├── raw/                    # Original HME100K/IAM images
│   └── processed/              # ground_truth.json and vocab.json
├── checkpoints/                # [LOCAL ONLY] Stored .pt weights per epoch
├── logs/                       # TensorBoard event logs (Loss/LR tracking)
├── notebooks/                  # Experimental scripts and testing
├── train.py                    # Main Blackwell-optimized training script
├── sanity_check.py             # CUDA/Dimension verification script
├── .gitignore                  # Filter for massive datasets/checkpoints
└── README.md                   # Project documentation
