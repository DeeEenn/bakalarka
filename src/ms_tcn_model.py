import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, channels, dropout=0.5):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim_in, num_classes, dropout=0.5):
        super().__init__()
        self.conv_in = nn.Conv1d(dim_in, num_f_maps, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(dilation=2 ** i, channels=num_f_maps, dropout=dropout)
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MSTCN(nn.Module):
    """
    Multi-Stage TCN:
    - Stage 1 bere puvodni feature vstup
    - Dalsi stage berou softmax vystup predchoziho stage a rafinuji predikci
    """

    def __init__(
        self,
        num_stages,
        num_layers,
        num_f_maps,
        dim_in,
        num_classes,
        dropout=0.5,
    ):
        super().__init__()

        self.stage1 = SingleStageTCN(
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            dim_in=dim_in,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.stages = nn.ModuleList(
            [
                SingleStageTCN(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    dim_in=num_classes,  # refinement stage vstup
                    num_classes=num_classes,
                    dropout=dropout,
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x):
        outputs = []

        out = self.stage1(x)
        outputs.append(out)

        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)

        # Vraci list/stoh vsech stage logits: [S, B, C, T]
        return torch.stack(outputs, dim=0)