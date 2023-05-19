import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms.functional as TF

from ml_collections import ConfigDict

import math


def Resnet34Backbone(num_layers, max_pool_stride_1=True, pretrained=True):
    m = tv.models.resnet34(pretrained=pretrained)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    if max_pool_stride_1:
        first_max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    else:
        first_max_pool = m.maxpool
    blocks = [input_conv, m.bn1, m.relu, first_max_pool]
    for i in range(1, num_layers + 1):
        blocks.append(m.__getattr__(f"layer{i}"))
    model = nn.Sequential(*blocks)
    return model


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.out_features = hidden_size * 2
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, in_features: int, dropout: float = 0.1, max_len: int = 1500):
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

    def __init__(self, in_features, num_layers, num_heads=4, pe_max_len: int = 1500):
        super(TransformerEncoder, self).__init__()
        self.pe = PositionalEncoding(in_features, dropout=0.1, max_len=pe_max_len)
        layer = nn.TransformerEncoderLayer(in_features, num_heads, dim_feedforward=in_features, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer,num_layers=num_layers)
        
        self.out_features = in_features

    def forward(self, x):
        return self.encoder(self.pe(x))


class CTCDecoderModel(nn.Module):

    def __init__(self, backbone, encoder, time_feature_count, num_classes):
        super(CTCDecoderModel, self).__init__()
        
        self.backbone = backbone
        
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        
        self.encoder = encoder
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_features, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ParallelModel(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList([*models])

    def __len__(self):
        return len(self.models)

    def forward(self, *args, **kwargs):
        results = [model(*args, **kwargs) for model in self.models]
        return tuple(results)
    
    
def make_single_model_v2(config: ConfigDict):
    
    backbone_constructor = config.backbone.constructor
    backbone_params = config.backbone[backbone_constructor]
    
    encoder_constructor = config.encoder.constructor
    encoder_params = config.encoder[encoder_constructor]
    
    backbone = eval(backbone_constructor)(**backbone_params)
    encoder = eval(encoder_constructor)(**encoder_params)
    
    decoder_constructor = config.decoder.constructor
    decoder_params = config.decoder[decoder_constructor]
    
    model = eval(decoder_constructor)(backbone, encoder, **decoder_params)
    
    return model
    

def make_model_v2(config: ConfigDict):
    if config.type == "single":
        return make_single_model_v2(config.first)
    elif config.type == "duo":
        first = make_single_model_v2(config.first)
        second = make_single_model_v2(config.second)
        return ParallelModel(first, second)
    
    
        