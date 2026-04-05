# 🚀 LaTeX Converter: 101M Parameter Hybrid Math OCR

> **⚠️ WORK IN PROGRESS (WIP)** > This project is currently in active development. Training is ongoing on an NVIDIA RTX 5070 Ti (Blackwell). Current loss and model weights are subject to significant change.

A state-of-the-art Mathematical Expression Recognition (MER) system that translates handwritten math and text images into precise LaTeX strings. This project combines deep convolutional feature extraction with the sequential power of Transformers.

---

## 🧠 Architecture Overview

The model employs a **Hybrid CNN-Transformer** architecture designed to handle the multi-scale nature of mathematical notation.

1.  **Encoder (Vision):** A modified **ResNet-50** backbone. We utilize the feature maps before the final pooling layer to preserve spatial resolution.
2.  **Spatial Encoding:** 2D Sinusoidal Positional Encodings are injected into the visual features to maintain the relative coordinates of mathematical symbols.
3.  **Decoder (Language):** An **8-layer Transformer Decoder** ($d_{model}=768, n_{head}=8$) that autoregressively predicts LaTeX tokens using causal masking.

### Objective Function
The model is trained using Cross-Entropy Loss with Label Smoothing ($\alpha=0.1$):
$$\mathcal{L} = -\sum_{c=1}^{V} y_c \log(\hat{y}_c)$$

---

## ⚡ Hardware Optimization

This implementation is specifically tuned for **NVIDIA Blackwell (sm_120)** architecture:
- **Mixed Precision:** Uses `BFloat16` for superior numerical stability in Transformer training.
- **Acceleration:** Leverages `torch.compile` (Inductor) to fuse kernels and maximize Tensor Core utilization.
- **Compute:** Optimized for the **RTX 5070 Ti** Laptop GPU.

---

## 📂 Project Structure

```text
.
├── src/
│   ├── model.py            # 101M Hybrid Architecture logic
│   ├── tokenizer.py        # Custom LaTeX Vocabulary & Encoding
│   ├── dataset.py          # Data pipeline and batching
│   └── augmentations.py    # Albumentations (Elastic, Grid, Morph)
├── checkpoints/            # [Local] Saved .pt model weights
├── logs/                   # [Local] TensorBoard event logs
├── data/                   # [Local] HME100K & IAM datasets
├── train.py                # Main training engine
├── sanity_check.py         # Dimension and CUDA verification
└── README.md
