import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import torch
from torch.nn import init

BN = partial(nn.BatchNorm1d, eps=5e-3,momentum=0.1)

class StandardAttentionTransformerEncoder(nn.Module):
    def __init__(
            self,
            dim_model,
            num_layers,
            num_heads=None,
            dim_feedforward=None,
            dropout=0.0,
            norm_first=False,
            activation=F.gelu,
    ):
        super().__init__()

        if num_heads is None:
            num_heads = dim_model // 64

        if dim_feedforward is None:
            dim_feedforward = dim_model * 4

        if isinstance(activation, str):
            activation = {
                'relu': F.relu,
                'gelu': F.gelu
            }.get(activation)

            if activation is None:
                raise ValueError(f'Unknown activation {activation}')

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=1e-6,
            norm_first=norm_first,
            batch_first=True  # Ensures that input is (batch, seqlen, dim_model)
        )

        self.layers = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x should be of shape (batch, seqlen, dim_model)
        x = self.layers(x, src_key_padding_mask=src_key_padding_mask)
        return x

"Squeeze-and-Excitation Networks"

class SEAttention(nn.Module):

    def __init__(self, channel=3,reduction=16):
        super().__init__()
  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out
    

class Conv1dBnRelu(nn.Module):
	def __init__(self, in_channel,out_channel,kernel_size,stride=1,padding=0, is_bn=True):
		super().__init__()
		self.is_bn = is_bn
		self.conv=nn.Conv1d(
			in_channel, out_channel, kernel_size=kernel_size,
			stride=stride, padding=padding, bias=not is_bn
		)
		self.bn=BN(out_channel)

	def forward(self, x):
		x=self.conv(x)
		if self.is_bn:
			x=self.bn(x)
		x = F.relu(x,inplace=True)
		return x