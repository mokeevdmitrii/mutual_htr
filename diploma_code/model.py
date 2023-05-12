import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import typing as tp

from ml_collections import ConfigDict

from .data_loader.transforms import HorizontalChunker

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', activ=nn.ReLU6, norm=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm(num_features=out_channels)
        self.activ = activ()

    def forward(self, x):
        return self.activ(self.norm(self.conv(x)))


class FusedInvertedBottleneck(nn.Module):

    def __init__(self, in_channels: int, scale_factor: int):
        super(FusedInvertedBottleneck, self).__init__()
        self.ex_conv = ConvBlock(in_channels, in_channels * scale_factor, kernel_size=3)
        self.sh_conv = ConvBlock(in_channels * scale_factor, in_channels, kernel_size=1, activ=nn.Identity)

    def forward(self, x):
        y = self.ex_conv(x)
        y = self.sh_conv(y)
        return x + y


class ReduceBlock(nn.Module):

    def __init__(self, in_channels, exp_factor, k_1, k_2, k_3, k_4):
        super(ReduceBlock, self).__init__()

        if (k_1 + k_3 - k_4) != 1:
            raise ValueError(f"{k_1} + {k_3} - {k_4} != 1")
        self.l1 = ConvBlock(in_channels, in_channels, kernel_size=(1,1))
        self.l2 = ConvBlock(in_channels, in_channels, kernel_size=(k_1,1),
                            padding='valid')
        self.l3 = ConvBlock(in_channels, in_channels, kernel_size=(1,k_2))
        self.l4 = ConvBlock(in_channels, exp_factor * in_channels,
                            kernel_size=(k_3,1), padding='valid')
        self.r1 = ConvBlock(in_channels, exp_factor * in_channels,
                            kernel_size=(k_4,1),padding='valid')

    def forward(self, x):
        left = self.l4(self.l3(self.l2(self.l1(x))))
        right = self.r1(x)
        return torch.cat([left, right], dim=1)



class Backbone(nn.Module):

    def __init__(self, out_features):
        super(Backbone, self).__init__()

        assert out_features % 4 == 0, f"assert {out_features} % 4 == 0"

        self.c_1 = ConvBlock(16, out_features // 2, kernel_size=(3,3))
        self.c_2 = ConvBlock(out_features // 2, out_features // 4, kernel_size=(1,1))
        self.bns = nn.ModuleList([FusedInvertedBottleneck(out_features // 4, scale_factor=8) for i in range(10)])
        self.reduce = ReduceBlock(out_features // 4, exp_factor=2, k_1=5, k_2=5, k_3=6, k_4=10)

    def forward(self, x):
        y = F.pixel_unshuffle(x, 4)
        y = self.c_1(y)
        y = self.c_2(y)
        for b in self.bns:
            y = b(y)
        y = self.reduce(y)
        # batch seq feature
        return y.squeeze(2).permute(0, 2, 1)


class PositionalEncoding(nn.Module):

    def __init__(self, in_features: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_features, 2) * (-math.log(10000.0) / in_features))
        pe = torch.zeros(1, max_len, in_features)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('weight', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.weight[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):

    def __init__(self, in_features, num_layers, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self.pe = PositionalEncoding(in_features, dropout=0.1, max_len=1000)
        layer = nn.TransformerEncoderLayer(in_features, num_heads, dim_feedforward=in_features, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer,num_layers=num_layers)

    def forward(self, x):
        return self.encoder(self.pe(x))


class CTCRawDecoder(nn.Module):

    def __init__(self, in_features, vocab_size):
        """
        Note: vocab size is given with blank character already
        """
        super(CTCRawDecoder, self).__init__()

        self.linear = nn.Linear(in_features, vocab_size)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.out(self.linear(x))


class CTCDecoderModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, encoder: torch.nn.Module, decoder_config: ConfigDict):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = CTCRawDecoder(decoder_config.in_features, decoder_config.out_features)
        self.chunker = HorizontalChunker(decoder_config.image_height, decoder_config.chunk_size)

    def forward(self, batch: tp.Dict[str, tp.Any]):
        inputs, indexes, widths = batch['input'], batch['idx'], batch['w']

        outputs = self.backbone(inputs)

        outputs = self.encoder(outputs)

        W_orig = inputs.size(3)
        W_shrinked = outputs.size(1)

        assert W_orig % W_shrinked == 0, f"assert {W_orig} % {W_shrinked} == 0"

        features_widths = torch.div(widths, W_orig // W_shrinked, rounding_mode="floor")

        outputs, chunk_lens = self.chunker.merge(outputs, indexes, features_widths)

        outputs = self.decoder(outputs)

        return outputs, chunk_lens


class ParallelModel(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList([*models])

    def __len__(self):
        return len(self.models)

    def forward(self, *args, **kwargs):
        results = [model(*args, **kwargs) for model in self.models]
        return tuple(t for t in zip(*results))


def make_single_model(config: ConfigDict):
    backbone = Backbone(config.backbone.out_features)

    encoder_type = config.encoder.type
    if encoder_type == "attn":
        encoder = TransformerEncoder(config.encoder.features, **config.encoder.attn)
    else:
        raise ValueError(f"unsupported encoder type: {encoder_type}")

    decoder_type = config.decoder.type
    if decoder_type == "ctc":
        model = CTCDecoderModel(backbone, encoder, config.decoder.ctc)
        return model
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")


def make_model(config: ConfigDict):
    if config.type == "single":
        return make_single_model(config.first)
    elif config.type == "duo":
        first = make_single_model(config.first)
        second = make_single_model(config.second)
        return ParallelModel(first, second)



