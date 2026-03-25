import math
from typing import Optional

import torch
import torch.nn as nn


class TemporalConvFFN(nn.Module):
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        y = x.transpose(1, 2)
        y = self.dw_conv(y)
        y = self.pw_in(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.pw_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class ASFormerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        dilation: int,
        window_size: int,
    ):
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
        self.window_size = max(1, window_size)

    def _build_local_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = torch.abs(idx[:, None] - idx[None, :])
        mask = torch.zeros((seq_len, seq_len), device=device)
        return mask.masked_fill(dist > self.window_size, float("-inf"))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask
        local_attn_mask = self._build_local_attention_mask(x.size(1), x.device)

        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=local_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_attn(attn_out)

        ffn_in = self.norm_ffn(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.drop_ffn(ffn_out)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class ASFormer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        input_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_dilation: int = 16,
        max_window: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.input_dropout = nn.Dropout(dropout)

        layers = []
        for i in range(num_layers):
            dilation = min(2 ** i, max_dilation)
            window_size = min(2 ** i, max_window)
            layers.append(
                ASFormerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    dilation=dilation,
                    window_size=window_size,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def _sinusoidal_positional_encoding(
        length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        if mask is None:
            mask = torch.ones((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)

        pe = self._sinusoidal_positional_encoding(
            length=x.size(1),
            dim=x.size(2),
            device=x.device,
            dtype=x.dtype,
        )
        x = self.input_dropout(x + pe)
        x = x * mask.unsqueeze(-1).to(x.dtype)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits.transpose(1, 2)
