"""
Multi-task MS-TCN for inhaler technique recognition.

Outputs:
1. Frame-level phase predictions (6 classes: KLID, PRIPRAVA, ROZDEJCHANI, INHALACE, ZADRZENI, WYDECH)
2. Video-level error type classification (9 classes)
3. Video-level error step classification (5 classes)
4. Video-level correctness binary classification
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, channels, dropout=0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = self.norm(x)
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = x + out
        if mask is not None:
            out = out * mask
        return out


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_f_maps,
        dim_in,
        num_classes,
        dropout=0.5,
        max_dilation=512,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(dim_in, num_f_maps, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    dilation=min(2 ** i, max_dilation),
                    channels=num_f_maps,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        out = self.conv_in(x)
        if mask is not None:
            out = out * mask
        for layer in self.layers:
            out = layer(out, mask=mask)
        out = self.conv_out(out)
        if mask is not None:
            out = out * mask
        return out


class MSTCNMultitask(nn.Module):
    """
    Multi-task MS-TCN with auxiliary heads for error classification.
    
    Architecture:
    - Shared encoder: Multi-stage TCN for temporal segmentation
    - Main task: Frame-level phase classification (from last stage)
    - Auxiliary tasks:
        - Error type classification (video-level)
        - Error step classification (video-level)
        - Correctness classification (video-level, binary)
    """
    def __init__(
        self,
        num_stages: int,
        num_layers: int,
        num_f_maps: int,
        dim_in: int,
        num_phases: int = 6,
        num_error_types: int = 9,
        num_error_steps: int = 5,
        dropout: float = 0.5,
        max_dilation: int = 512,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_f_maps = num_f_maps

        # Main MS-TCN stages
        self.stage1 = SingleStageTCN(
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            dim_in=dim_in,
            num_classes=num_phases,
            dropout=dropout,
            max_dilation=max_dilation,
        )

        self.stages = nn.ModuleList(
            [
                SingleStageTCN(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    dim_in=num_phases,
                    num_classes=num_phases,
                    dropout=dropout,
                    max_dilation=max_dilation,
                )
                for _ in range(num_stages - 1)
            ]
        )
        
        # Auxiliary heads for video-level classification
        # Extract features from the last stage before final classification
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Error type head
        self.error_type_head = nn.Sequential(
            nn.Linear(num_f_maps, num_f_maps // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_f_maps // 2, num_error_types)
        )
        
        # Error step head
        self.error_step_head = nn.Sequential(
            nn.Linear(num_f_maps, num_f_maps // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_f_maps // 2, num_error_steps)
        )
        
        # Correctness head (binary)
        self.correctness_head = nn.Sequential(
            nn.Linear(num_f_maps, num_f_maps // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_f_maps // 2, 2)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-task learning.
        
        Args:
            x: Input features (B, C, T)
            mask: Valid frame mask (B, T)
        
        Returns:
            phase_logits: Frame-level phase predictions (B, num_phases, T) - from last stage
            error_type_logits: Video-level error type predictions (B, num_error_types)
            error_step_logits: Video-level error step predictions (B, num_error_steps)
            correctness_logits: Video-level correctness predictions (B, 2)
        """
        mask_1d = None
        if mask is not None:
            mask_1d = mask.unsqueeze(1).to(x.dtype)

        # Run multi-stage refinement (main task)
        out = self.stage1(x, mask=mask_1d)
        stage_outputs = [out]

        for stage in self.stages:
            out = stage(F.softmax(out, dim=1), mask=mask_1d)
            stage_outputs.append(out)
        
        # Phase predictions from last stage
        phase_logits = stage_outputs[-1]  # (B, num_phases, T)
        
        # Extract features for auxiliary tasks
        # Use intermediate features from stage1 before final classification
        # We need to re-extract features from input
        features = self.stage1.conv_in(x)  # (B, num_f_maps, T)
        if mask_1d is not None:
            features = features * mask_1d
        
        for layer in self.stage1.layers:
            features = layer(features, mask=mask_1d)
        
        # Global pooling over time (masked average)
        if mask is not None:
            features_masked = features * mask_1d
            mask_sum = mask.sum(dim=1, keepdim=True).unsqueeze(1).clamp(min=1)  # (B, 1, 1)
            features_pooled = features_masked.sum(dim=2) / mask_sum.squeeze(1)  # (B, num_f_maps)
        else:
            features_pooled = features.mean(dim=2)  # (B, num_f_maps)
        
        # Video-level classification
        error_type_logits = self.error_type_head(features_pooled)  # (B, num_error_types)
        error_step_logits = self.error_step_head(features_pooled)  # (B, num_error_steps)
        correctness_logits = self.correctness_head(features_pooled)  # (B, 2)
        
        return phase_logits, error_type_logits, error_step_logits, correctness_logits
