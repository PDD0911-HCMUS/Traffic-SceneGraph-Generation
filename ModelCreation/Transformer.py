# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512):
        super().__init__()
        nhead=8
        num_encoder_layers=6
        num_decoder_layers=6
        dim_feedforward=2048
        dropout=0.1
        activation="relu"
        normalize_before=False
        return_intermediate_dec=True

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        # cp_tgt = torch.zeros_like(query_embed)
        '''
        Note: 
        Because we use torch.zeros -> cp_tgt will be on CPU
        '''
        cp_tgt = torch.zeros(query_embed.size(0), query_embed.size(1), query_embed.size(2)*2).to(query_embed.device)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # hs_rel, hs_sub, hs_obj = self.decoder(sub_entity, obj_entity, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, sub_pos=sub_embed, obj_pos = obj_embed)
        hs= self.decoder(cp_tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return  hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, cp_tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        output = cp_tgt

        intermediate = []


        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)


        if self.return_intermediate:
            return torch.stack(intermediate)
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        """Get activation function for model"""
        self.activation = _get_activation_fn(activation)

        '''Couple object section'''
        self.self_attn_sub = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # self-attention
        self.multihead_attn_sub = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # cross attention with output encoder
        # Implementation of Feedforward model
        self.linear1_sub = nn.Linear(d_model, dim_feedforward)
        self.dropout_sub = nn.Dropout(dropout)
        self.linear2_sub = nn.Linear(dim_feedforward, d_model)

        self.norm1_sub = nn.LayerNorm(d_model)
        self.norm2_sub = nn.LayerNorm(d_model)
        self.norm3_sub = nn.LayerNorm(d_model)
        self.dropout1_sub = nn.Dropout(dropout)
        self.dropout2_sub = nn.Dropout(dropout)
        self.dropout3_sub = nn.Dropout(dropout)

        self.self_attn_obj = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # self-attention
        self.multihead_attn_obj = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # cross attention with output encoder
        # Implementation of Feedforward model
        self.linear1_obj = nn.Linear(d_model, dim_feedforward)
        self.dropout_obj = nn.Dropout(dropout)
        self.linear2_obj = nn.Linear(dim_feedforward, d_model)

        self.norm1_obj = nn.LayerNorm(d_model)
        self.norm2_obj = nn.LayerNorm(d_model)
        self.norm3_obj = nn.LayerNorm(d_model)
        self.dropout1_obj = nn.Dropout(dropout)
        self.dropout2_obj = nn.Dropout(dropout)
        self.dropout3_obj = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_cp,
                memory, tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, 
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        h_dim = query_pos.shape[2]
        tgt_sub, tgt_obj = torch.split(tgt_cp, h_dim, dim=-1)
        '''Couple Entities Detection'''
        q_sub = k_sub = self.with_pos_embed(tgt_sub, query_pos)
        tgt2_sub = self.self_attn_sub(q_sub, k_sub, value=tgt_sub, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt_sub = tgt_sub + self.dropout1_sub(tgt2_sub)
        tgt_sub = self.norm1_sub(tgt_sub)
        
        tgt2_sub = self.multihead_attn_sub(query=self.with_pos_embed(tgt_sub, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt_sub = tgt_sub + self.dropout2_sub(tgt2_sub)
        tgt_sub = self.norm2_sub(tgt_sub)
        tgt2_sub = self.linear2_sub(self.dropout_sub(self.activation(self.linear1_sub(tgt_sub))))
        tgt_sub = tgt_sub + self.dropout3_sub(tgt2_sub)
        tgt_sub = self.norm3_sub(tgt_sub)

        
        q_obj = k_obj = self.with_pos_embed(tgt_sub, query_pos)
        tgt2_obj = self.self_attn_obj(q_obj, k_obj, value=tgt_obj, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt_obj = tgt_obj + self.dropout1_obj(tgt2_obj)
        tgt_obj = self.norm1_obj(tgt_obj)
        tgt2_obj = self.multihead_attn_obj(query=self.with_pos_embed(tgt_obj, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt_obj = tgt_obj + self.dropout2_obj(tgt2_obj)
        tgt_obj = self.norm2_obj(tgt_obj)
        tgt2_obj = self.linear2_obj(self.dropout_obj(self.activation(self.linear1_obj(tgt_obj))))
        tgt_obj = tgt_obj + self.dropout3_obj(tgt2_obj)
        tgt_obj = self.norm3_obj(tgt_obj)

        # tgt_rel = tgt_sub + tgt_obj
        # return tgt, cp_maps
        tgt_cp = torch.cat((tgt_sub, tgt_obj), dim=-1)
        return tgt_cp
        #return tgt_sub, tgt_obj
        #return tgt_sub, tgt_att, cp_maps


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(d_model):
    return Transformer(d_model=d_model)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")