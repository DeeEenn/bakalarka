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


class MSTCN(nn.Module):
    def __init__(
        self,
        num_stages,
        num_layers,
        num_f_maps,
        dim_in,
        num_classes,
        dropout=0.5,
        max_dilation=512,
    ):
        super().__init__()

        self.stage1 = SingleStageTCN(
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            dim_in=dim_in,
            num_classes=num_classes,
            dropout=dropout,
            max_dilation=max_dilation,
        )

        self.stages = nn.ModuleList(
            [
                SingleStageTCN(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    dim_in=num_classes,
                    num_classes=num_classes,
                    dropout=dropout,
                    max_dilation=max_dilation,
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask=None):
        outputs = []

        mask_1d = None
        if mask is not None:
            mask_1d = mask.unsqueeze(1).to(x.dtype)

        out = self.stage1(x, mask=mask_1d)
        outputs.append(out)

        for stage in self.stages:
            out = stage(F.softmax(out, dim=1), mask=mask_1d)
            outputs.append(out)

        return torch.stack(outputs, dim=0)
