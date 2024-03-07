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

    def forward(self, src, mask, cp_embed, att_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        cp_embed = cp_embed.unsqueeze(1).repeat(1, bs, 1)
        att_embed = att_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        cp_entities = torch.zeros_like(cp_embed)
        att_entity = torch.zeros_like(att_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, hs_att, hs_map = self.decoder(cp_entities, att_entity, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, cp_pos=cp_embed, att_pos = att_embed)
        return hs.transpose(1, 2), hs_att.transpose(1, 2), hs_map.reshape(hs_map.shape[0], bs, hs_map.shape[2], 1, h, w) \
            , memory.permute(1, 2, 0).view(bs, c, h, w)


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

    def forward(self, cp_entities, att_entity, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                cp_pos: Optional[Tensor] = None,
                att_pos: Optional[Tensor] = None):
        
        output_cp = cp_entities
        ouput_att = att_entity

        intermediate_cp = []
        intermediate_att = []
        intermediate_fmap = []


        for layer in self.layers:
            output_cp, ouput_att, cp_maps = layer(output_cp, ouput_att, cp_pos, att_pos, 
                                                  memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate_cp.append(output_cp)
                intermediate_att.append(ouput_att)
                intermediate_fmap.append(cp_maps)


        if self.return_intermediate:
            return torch.stack(intermediate_cp),torch.stack(intermediate_att),torch.stack(intermediate_fmap)

        #return output.unsqueeze(0)


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
        self.self_attn_cp = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # self-attention
        self.multihead_attn_cp = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # cross attention with output encoder
        # Implementation of Feedforward model
        self.linear1_cp = nn.Linear(d_model, dim_feedforward)
        self.dropout_cp = nn.Dropout(dropout)
        self.linear2_cp = nn.Linear(dim_feedforward, d_model)

        self.norm1_cp = nn.LayerNorm(d_model)
        self.norm2_cp = nn.LayerNorm(d_model)
        self.norm3_cp = nn.LayerNorm(d_model)
        self.dropout1_cp = nn.Dropout(dropout)
        self.dropout2_cp = nn.Dropout(dropout)
        self.dropout3_cp = nn.Dropout(dropout)

        '''Attribuite object section'''
        self.self_attn_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1_att = nn.Linear(d_model, dim_feedforward)
        self.dropout_att = nn.Dropout(dropout)
        self.linear2_att = nn.Linear(dim_feedforward, d_model)

        self.norm1_att = nn.LayerNorm(d_model)
        self.norm2_att = nn.LayerNorm(d_model)
        self.norm3_att = nn.LayerNorm(d_model)
        self.dropout1_att = nn.Dropout(dropout)
        self.dropout2_att = nn.Dropout(dropout)
        self.dropout3_att = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_cp, tgt_att, query_pos_cp, query_pos_att,
                memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        
        '''Couple object layer'''
        q_cp = k_cp = self.with_pos_embed(tgt_cp, query_pos_cp)
        tgt2_cp = self.self_attn_cp(q_cp, k_cp, value=tgt_cp, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt_cp = tgt_cp + self.dropout1_cp(tgt2_cp)
        tgt_cp = self.norm1_cp(tgt_cp)
        tgt2_cp, cp_maps = self.multihead_attn_cp(query=self.with_pos_embed(tgt_cp, query_pos_cp),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt_cp = tgt_cp + self.dropout2_cp(tgt2_cp)
        tgt_cp = self.norm2_cp(tgt_cp)
        tgt2_cp = self.linear2_cp(self.dropout_cp(self.activation(self.linear1_cp(tgt_cp))))
        tgt_cp = tgt_cp + self.dropout3_cp(tgt2_cp)
        tgt_cp = self.norm3_cp(tgt_cp)

        '''Attribute object layer'''
        q_att = k_att = self.with_pos_embed(tgt_att, query_pos_att)
        tgt2_att = self.self_attn_att(q_att, k_att, value=tgt_att, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt_att = tgt_att + self.dropout1_att(tgt2_att)
        tgt_att = self.norm1_att(tgt_att)
        tgt2_att = self.multihead_attn_att(query=self.with_pos_embed(tgt_att, query_pos_att),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt_att = tgt_att + self.dropout2_att(tgt2_att)
        tgt_att = self.norm2_att(tgt_att)
        tgt2_att = self.linear2_att(self.dropout_att(self.activation(self.linear1_att(tgt_att))))
        tgt_att = tgt_att + self.dropout3_att(tgt2_att)
        tgt_att = self.norm3_att(tgt_att)
        return tgt_cp, tgt_att, cp_maps


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