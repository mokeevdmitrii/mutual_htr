import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms.functional as TF

from ml_collections import ConfigDict

import typing as tp

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

    
    
class CoolTransformerEncoderLayer(nn.Module):
    
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: tp.Union[str, tp.Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
            
    
    def forward(self,
                src: torch.Tensor,
                src_mask: tp.Optional[torch.Tensor] = None,
                src_key_padding_mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: tp.Optional[torch.Tensor], 
                  key_padding_mask: tp.Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class TransformerEncoder(nn.Module):

    def __init__(self, in_features, hidden_features, num_layers, num_heads=4, pe_max_len: int = 1500):
        super(TransformerEncoder, self).__init__()
        self.pe = PositionalEncoding(in_features, dropout=0.1, max_len=pe_max_len)
        #layer = nn.TransformerEncoderLayer(in_features, num_heads, dim_feedforward=hidden_features, dropout=0.1, batch_first=True)
        layer = CoolTransformerEncoderLayer(in_features, num_heads, dim_feedforward=hidden_features, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer,num_layers=num_layers)
        
        self.out_features = in_features
        
        with torch.no_grad():
            self.reset_parameters()
        
    def reset_parameters(self):
        N = len(self.encoder.layers)
        N_1_4 = N ** -0.25
        for mn, m in self.encoder.named_modules():
            if isinstance(m, (torch.nn.MultiheadAttention, torch.nn.Linear)):
                for pn, p in m.named_parameters():
                    if 'weight' in pn:
                        torch.nn.init.xavier_uniform_(p)
                        p = torch.mul(p, 0.67 * N_1_4)
                    elif 'bias' in pn:
                        torch.nn.init.zeros_(p)
            elif isinstance(m, (torch.nn.LayerNorm, )):
                for pn, p in m.named_parameters():
                    if 'weight' in pn:
                        torch.nn.init.ones_(p)
                    elif 'bias' in pn:
                        torch.nn.init.zeros_(p)
                

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
    elif str(config.type).startswith("duo"):
        first = make_single_model_v2(config.first)
        second = make_single_model_v2(config.second)
        return ParallelModel(first, second)
    
    
        