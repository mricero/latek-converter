import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=150):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(d_model, max_h, max_w)
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        
        pos_w = torch.arange(0, max_w).unsqueeze(1)
        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        
        pos_h = torch.arange(0, max_h).unsqueeze(1)
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(2), :x.size(3)]

class HybridMathOCR(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=8):
        super().__init__()
        
        # 1. Vision Encoder
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.bottleneck = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # 2. Spatial Encodings
        self.pos_encoder_2d = PositionalEncoding2D(d_model)
        
        # FIX: Shape changed to (Max_Seq, 1, d_model) to match Transformer Decoder flow
        self.pos_encoder_1d = nn.Parameter(torch.randn(1000, 1, d_model)) 
        
        # 3. Transformer Decoder
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # Increased dim_feedforward to 3072 for better capacity
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=3072,
            activation='gelu' # GELU is standard for modern transformers
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, images, targets, tgt_mask=None):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
            
        features = self.backbone(images) 
        features = self.bottleneck(features) 
        features = self.pos_encoder_2d(features)
        
        B, C, H, W = features.shape
        memory = features.view(B, C, -1).permute(2, 0, 1)
        
        tgt_emb = self.text_embedding(targets).permute(1, 0, 2)
        tgt_emb = tgt_emb + self.pos_encoder_1d[:tgt_emb.size(0), :, :]
        
        # FIX: Explicitly set is_causal=True to avoid the graph break check
        output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_is_causal=True if tgt_mask is not None else False
        )
        return self.fc_out(output)