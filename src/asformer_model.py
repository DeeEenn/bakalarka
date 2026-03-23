import math
from typing import Optional

import torch
import torch.nn as nn


class TemporalConvFFN(nn.Module):
    """
    Feed-forward cast ASFormer vrstvy:
    depthwise dilatovana conv + pointwise projekce.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        ff_dim = d_model * 4

        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=d_model,
        )
        self.pw_in = nn.Conv1d(d_model, ff_dim, kernel_size=1)
        self.pw_out = nn.Conv1d(ff_dim, d_model, kernel_size=1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = x.transpose(1, 2)  # (B, C, T)
        y = self.dw_conv(y)
        y = self.pw_in(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.pw_out(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)  # (B, T, C)
        return y


class ASFormerLayer(nn.Module):
    """
    Jedna ASFormer vrstva:
    1) MHSA (self-attention)
    2) temporal FFN s dilatovanou konvoluci
    Oboji v pre-norm residual stylu.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float, dilation: int):
        super().__init__()

        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = TemporalConvFFN(d_model=d_model, dropout=dropout, dilation=dilation)
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C)
        # mask: (B, T) True = valid pozice, False = padding
        key_padding_mask = None if mask is None else ~mask

        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_attn(attn_out)

        ffn_in = self.norm_ffn(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.drop_ffn(ffn_out)
        return x


class ASFormer(nn.Module):
    """
    ASFormer-like model pro temporalni segmentaci:
    input projection -> stack [self-attention + temporal conv ffn] -> class head
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        input_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_dilation: int = 16,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.input_dropout = nn.Dropout(dropout)

        layers = []
        for i in range(num_layers):
            dilation = min(2 ** i, max_dilation)
            layers.append(
                ASFormerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    dilation=dilation,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def _sinusoidal_positional_encoding(
        length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # (1, T, C)
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, input_dim, T)
        x = self.input_proj(x)       # (B, d_model, T)
        x = x.transpose(1, 2)        # (B, T, d_model)

        pe = self._sinusoidal_positional_encoding(
            length=x.size(1),
            dim=x.size(2),
            device=x.device,
            dtype=x.dtype,
        )
        x = self.input_dropout(x + pe)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.classifier(x)   
        logits = logits.transpose(1, 2) 
        return logits