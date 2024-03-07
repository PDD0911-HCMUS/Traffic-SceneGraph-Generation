# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from Datasets.Util import (NestedTensor, nested_tensor_from_tensor_list)
from ModelCreation.BackboneCNN import build_backbone
# from .matcher import build_matcher
from ModelCreation.Transformer import build_transformer

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SGG(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """
    def __init__(self, backbone, num_couple, num_att, num_classes, transformer, num_rel
                # transformer , num_classes, num_rel_classes, num_entities, num_triplets, aux_loss=False, matcher=None
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.num_couple = num_couple
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 8, 3) #8 coordinates for 2 boxes

        #self.rel_class_embed = MLP(hidden_dim*2+128, hidden_dim, num_rel + 1, 2)


        self.cp_embed = nn.Embedding(num_couple, hidden_dim)
        self.att_embed = nn.Embedding(num_att, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.transformer = transformer
        
        
    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None
        #hs = self.transformer(self.input_proj(src), mask, self.cp_embed.weight, self.att_embed.weight, pos[-1])[0]
        hs, hs_att, hs_map, _ = self.transformer(self.input_proj(src), mask, self.cp_embed.weight, self.att_embed.weight, pos[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 
               'pred_boxes': outputs_coord[-1],
               #'rel_logits': outputs_class_rel[-1]
               }
        return out


def build():

    backbone = build_backbone()
    hidden_dim = 256
    transformer = build_transformer(d_model=hidden_dim)
    model = SGG(backbone, 
                transformer=transformer,
                num_couple = 100,
                num_att=100,
                num_classes=181,
                num_rel = 51
                )

    return model
