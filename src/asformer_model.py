import torch
import torch.nn as nn
import torch.nn.functional as F

class ASFormerBlock(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ASFormerBlock, self).__init__()
        # dilatovana konvoluce s filterem velikosti 3
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out # rezidualni spojeni pro stabilitu treninku
    
class ASFormer(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes):
        super(ASFormer, self).__init__()
        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)

        # hirearchicke okno pozornosti roustouci na 2 na i
        self.layers = nn.ModuleList([
            ASFormerBlock(2**i, num_f_maps, num_f_maps) for i in range(num_layers)
        ])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self,x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

