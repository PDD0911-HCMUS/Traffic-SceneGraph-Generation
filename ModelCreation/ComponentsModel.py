import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models._utils import IntermediateLayerGetter
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
#from PositionalEncoding import build_position_encoding
from ModelCreation.PositionalEncoding import build_position_encoding
from Datasets.Util import NestedTensor, nested_tensor_from_tensor_list, is_main_process
from typing import Dict, List

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b7')
        self.fc = nn.Linear(2560, 2048) #output dim from EFF-B7 is 2560

    def forward(self, x):
        features = self.cnn.extract_features(x)
        global_features = F.adaptive_avg_pool2d(features, 1)
        global_features = global_features.view(global_features.size(0), -1)
        global_features = self.fc(global_features)
        return global_features

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, 
                #  train_backbone: bool, num_channels: int, return_interm_layers: bool
                 ):
        super().__init__()
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        #self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = backbone
        self.num_channels = 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.item():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

class MyModel(nn.Module):
    def __init__(self, backBone):
        super(MyModel, self).__init__()

        self.backBone = backBone
    
    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backBone(samples)

        return features

def build_backbone():
    position_embedding = build_position_encoding()
    backboneEff = CNNBackbone()
    backbone = BackboneBase(backboneEff)
    model = Joiner(backbone, position_embedding)
    #model.num_channels = backbone.num_channels
    return model

def build_model():
    backbone = build_backbone()
    model = MyModel(backbone)
    return model

# if __name__ == '__main__':

#     model = build_backbone()

