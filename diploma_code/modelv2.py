import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms.functional as TF


def resnet_34_backbone(self, num_layers, pretrained=True):
    m = tv.models.resnet34(weights=tv.models.ResNet34_Weights.DEFAULT)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool]
    for i in range(1, num_layers + 1):
        blocks.append(m.__getattr__(f"layer{i}"))
    model = nn.Sequential(*blocks)
    return model


class BiLSTMEncoder:
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(BiLSTM, self).__init__()
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

    def __init__(self, in_features, num_layers, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self.pe = PositionalEncoding(in_features, dropout=0.1)
        layer = nn.TransformerEncoderLayer(in_features, num_heads, dim_feedforward=in_features, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer,num_layers=num_layers)
        
        self.out_features = in_features

    def forward(self, x):
        return self.encoder(self.pe(x))


class CTCDecoderModel(nn.Module):

    def __init__(self, backbone, encoder, time_feature_count, num_classes):
        super(RecognitionModel, self).__init__()
        
        self.backbone = backbone
        
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        
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
    
    
        