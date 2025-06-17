from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights

class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout_rate: float):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels=channels, out_channels=channels, kernel_size=3)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcite(channels=channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(x)
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)

class MiniResNet1D(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 128, dropout_rate: float = 0.15):
        super().__init__()
        # 1. 多分支卷積層，捕捉不同感受野的特徵
        self.branch3 = ConvBNReLU(in_channels=in_channels, out_channels=hidden_dim // 2, kernel_size=3)
        self.branch5 = ConvBNReLU(in_channels=in_channels, out_channels=hidden_dim // 2, kernel_size=5)
        self.branch7 = ConvBNReLU(in_channels=in_channels, out_channels=hidden_dim // 2, kernel_size=7)
        
        # 2. 融合層，將多分支特徵合併
        self.fuse = ConvBNReLU(in_channels=hidden_dim * 3 // 2, out_channels=hidden_dim, kernel_size=1)
        
        # 3. 堆疊的殘差區塊
        self.res_block1 = ResBlock(channels=hidden_dim, dropout_rate=dropout_rate)
        self.res_block2 = ResBlock(channels=hidden_dim, dropout_rate=dropout_rate)
        self.res_block3 = ResBlock(channels=hidden_dim, dropout_rate=dropout_rate)
        self.res_block4 = ResBlock(channels=hidden_dim, dropout_rate=dropout_rate)
        
        # 4. 全局平均池化，將時間維度壓縮成單一向量
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.fuse(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        return self.pool(x).squeeze(-1)

class CrossSwingAttn(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, attn_dropout: float = 0.15, max_len: int = 4000):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, swing_vectors: torch.Tensor, swing_mask: torch.Tensor) -> torch.Tensor:
        B, S, _ = swing_vectors.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, swing_vectors], dim=1)
        
        x = x + self.pos_embed[:, :S + 1]
        
        cls_mask = torch.zeros((B, 1), device=swing_vectors.device, dtype=torch.bool)
        key_padding_mask = torch.cat([cls_mask, ~swing_mask], dim=1)
        
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        attn_output = self.attn_dropout(attn_output)
        
        x = x + attn_output
        x = x + self.ffn(x)
        
        return x[:, 0]

class CoralLayer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes - 1)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.bias

def coral_prob(logits: torch.Tensor) -> torch.Tensor:
    cumulative_probs = torch.sigmoid(logits)
    left = torch.cat([torch.ones_like(cumulative_probs[:, :1]), cumulative_probs], dim=1)
    right = torch.cat([cumulative_probs, torch.zeros_like(cumulative_probs[:, :1])], dim=1)
    return torch.clamp(left - right, 0.0, 1.0)

class MultiHead(nn.Module):
    def __init__(self, input_dim: int, head_dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.gender = nn.Linear(input_dim, 2)
        self.hand = nn.Linear(input_dim, 2)
        self.years = CoralLayer(input_dim=input_dim, num_classes=3)
        self.level = CoralLayer(input_dim=input_dim, num_classes=4)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.dropout(features)
        return {
            'gender': self.gender(x),
            'hand': self.hand(x),
            'play_years': self.years(x),
            'level': self.level(x),
        }

class TableTennisBig(nn.Module):
    def __init__(
        self,
        hidden_dim:int=128,
        extra_dim:int=60,
        backbone_dropout: float=0.15,
        attn_dropout: float=0.15,
        head_dropout: float=0.25
    ):
        super().__init__()

        self.backbone = MiniResNet1D(
            in_channels=6, 
            hidden_dim=hidden_dim, 
            dropout_rate=backbone_dropout
        )
        self.cross_attn = CrossSwingAttn(
            embed_dim=hidden_dim, 
            num_heads=4, 
            attn_dropout=attn_dropout
        )
        self.use_extra = extra_dim > 0
        
        if self.use_extra:
            self.extra_bn = nn.BatchNorm1d(extra_dim)
        
        final_feature_dim = hidden_dim + (extra_dim if self.use_extra else 0)
        self.heads = MultiHead(
            input_dim=final_feature_dim, 
            head_dropout=head_dropout
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, S, C, T = feats.shape # (Batch, Swings, Channels, Time)
        
        feats_reshaped = feats.view(B * S, C, T)
        swing_vectors = self.backbone(feats_reshaped)
        swing_vectors = swing_vectors.view(B, S, -1)
        
        pooled_features = self.cross_attn(swing_vectors=swing_vectors, swing_mask=mask)
        
        if self.use_extra and extras is not None:
            avg_extras = self.extra_bn(extras.mean(dim=1))
            pooled_features = torch.cat([pooled_features, avg_extras], dim=1)
            
        return self.heads(pooled_features)

    def coral_loss(self, logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        labels = y
        device = labels.device
        label_matrix = (torch.arange(logits.size(1), device=device)[None, :] < labels[:, None]).float()
        
        return F.binary_cross_entropy_with_logits(logits, label_matrix, weight=weight, reduction='mean')