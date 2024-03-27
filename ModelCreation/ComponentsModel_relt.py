# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from Datasets.Util import (NestedTensor, nested_tensor_from_tensor_list, accuracy, generalized_box_iou, box_cxcywh_to_xyxy, is_dist_avail_and_initialized, get_world_size)
from ModelCreation.BackboneCNN import build_backbone
# from .matcher import build_matcher
from ModelCreation.Transformer import build_transformer
from ModelCreation.MatcherSub import build_matcher_sub
from ModelCreation.MatcherObj import build_matcher_obj

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
    def __init__(self, backbone, 
                 num_queries, 
                 num_classes, transformer, 
                 num_rel_classes, aux_loss=False
                # transformer , num_classes, num_rel_classes, num_entities, num_triplets, aux_loss=False, matcher=None
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_sub: number of sub queries
            num_obj: number of obj queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        
        
        ''' FFN for sub-obj detection'''
        self.class_sub_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.class_obj_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_sub_embed = MLP(hidden_dim, hidden_dim, 4, 3) 
        self.bbox_obj_embed = MLP(hidden_dim, hidden_dim, 4, 3) 

        ''' FFN for rel detection'''
        #self.rel_class_embed = MLP(hidden_dim + 128, hidden_dim, num_rel + 1, 2)
        #self.rel_class_embed = MLP(hidden_dim, hidden_dim, num_rel_classes + 1, 2)

        '''sub queries and obj queries'''
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        '''Backbone'''
        self.backbone = backbone

        self.aux_loss = aux_loss
        

        

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        hs,_ = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        print('hs.size(): ', hs.size())
        hs_sub, hs_obj = torch.split(hs, self.hidden_dim, dim=-1)
        outputs_sub_class = self.class_sub_embed(hs_sub)
        outputs_obj_class = self.class_obj_embed(hs_obj)
        outputs_coord_sub = self.bbox_sub_embed(hs_sub).sigmoid()
        outputs_coord_obj = self.bbox_obj_embed(hs_obj).sigmoid()

        #outputs_rel = self.rel_class_embed(torch.cat((hs, hs_map), dim=-1))
        #outputs_rel = self.rel_class_embed(hs_rel)

        out = {'pred_sub_logits': outputs_sub_class[-1],
               'pred_obj_logits': outputs_obj_class[-1], 
               'pred_boxes_sub': outputs_coord_sub[-1],
               'pred_boxes_obj': outputs_coord_obj[-1]
            #    'pred_rel': outputs_rel[-1]
               }
        #print('self.aux_loss: ', self.aux_loss)
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_sub_class, outputs_obj_class, outputs_coord_sub, outputs_coord_obj)

        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_sub_class, outputs_obj_class, outputs_coord_sub, outputs_coord_obj):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_sub_logits': a, 'pred_obj_logits': b, 'pred_boxes_sub': c, 'pred_boxes_obj': d}
                for a, b, c, d in zip(outputs_sub_class[:-1], outputs_obj_class[-1], outputs_coord_sub[:-1], outputs_coord_obj[-1])]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, num_rel_classes, matcher_sub, matcher_obj, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes= num_classes
        self.num_rel_classes = num_rel_classes
        self.matcher_sub = matcher_sub
        self.matcher_obj = matcher_obj
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight_sub = torch.ones(self.num_classes + 1)
        empty_weight_obj = torch.ones(self.num_classes + 1)
        # empty_weight_rel = torch.ones(self.num_rel_classes + 1)

        empty_weight_sub[-1] = self.eos_coef
        empty_weight_obj[-1] = self.eos_coef
        # empty_weight_rel[-1] = self.eos_coef

        

        self.register_buffer('empty_weight_sub', empty_weight_sub)
        self.register_buffer('empty_weight_obj', empty_weight_obj)
        # self.register_buffer('empty_weight_rel', empty_weight_rel)

    def loss_labels(self, outputs, targets, indices_sub, indices_obj, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        #assert 'pred_logits' in outputs

        src_logits_sub = outputs['pred_sub_logits']
        src_logits_obj = outputs['pred_obj_logits']

        idx_sub = self._get_src_permutation_idx(indices_sub)
        idx_obj = self._get_src_permutation_idx(indices_obj)

        target_classes_sub_o = torch.cat([t["sub"][J] for t, (_, J) in zip(targets, indices_sub)])
        target_classes_sub = torch.full(src_logits_sub.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits_sub.device)
        target_classes_sub[idx_sub] = target_classes_sub_o

        target_classes_obj_o = torch.cat([t["obj"][J] for t, (_, J) in zip(targets, indices_obj)])
        target_classes_obj = torch.full(src_logits_obj.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits_obj.device)
        target_classes_obj[idx_obj] = target_classes_obj_o

        loss_ce_sub = F.cross_entropy(src_logits_sub.transpose(1, 2), target_classes_sub, self.empty_weight_sub)
        loss_ce_obj = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes_obj, self.empty_weight_obj)

        losses = {'loss_ce_sub': loss_ce_sub}
        losses = {'loss_ce_obj': loss_ce_obj}
        # losses = {'loss_ce': loss_ce_sub + loss_ce_obj}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['sub_error'] = 100 - accuracy(src_logits_sub[idx_sub], target_classes_sub_o)[0]
            losses['obj_error'] = 100 - accuracy(src_logits_obj[idx_obj], target_classes_obj_o)[0]
        return losses
    
    def loss_boxes(self, outputs, targets, indices_sub, indices_obj, num_boxes_sub, num_boxes_obj):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        #assert 'pred_boxes' in outputs
        idx_sub = self._get_src_permutation_idx(indices_sub)
        idx_obj = self._get_src_permutation_idx(indices_obj)
        
        outputs_pred_boxes_sub = outputs['pred_boxes_sub']
        outputs_pred_boxes_obj = outputs['pred_boxes_obj']

        src_boxes_sub = outputs_pred_boxes_sub[idx_sub]
        src_boxes_obj = outputs_pred_boxes_obj[idx_obj]

        target_boxes_sub = torch.cat([t['subBbox'][i] for t, (_, i) in zip(targets, indices_sub)], dim=0)
        target_boxes_obj = torch.cat([t['objBbox'][i] for t, (_, i) in zip(targets, indices_obj)], dim=0)

        loss_bbox_sub = F.l1_loss(src_boxes_sub, target_boxes_sub, reduction='none')
        loss_bbox_obj = F.l1_loss(src_boxes_obj, target_boxes_obj, reduction='none')

        losses = {}
        losses['loss_bbox_sub'] = loss_bbox_sub.sum() / num_boxes_sub
        losses['loss_bbox_obj'] = loss_bbox_obj.sum() / num_boxes_obj
        # losses['loss_bbox'] = (loss_bbox_sub.sum() + loss_bbox_obj.sum()) / num_boxes
        # losses['loss_bbox'] = (loss_bbox_sub.sum() + loss_bbox_obj.sum())

        loss_giou_sub = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes_sub),
            box_cxcywh_to_xyxy(target_boxes_sub)))
        
        loss_giou_obj = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes_obj),
            box_cxcywh_to_xyxy(target_boxes_obj)))
        
        losses['loss_giou_sub'] = loss_giou_sub.sum() / num_boxes_sub
        losses['loss_giou_obj'] = loss_giou_obj.sum() / num_boxes_obj
        # losses['loss_giou'] = (loss_giou_sub.sum() + loss_giou_obj.sum()) / num_boxes
        # losses['loss_giou'] = losses['loss_giou_sub'] + losses['loss_giou_obj']

        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices_sub, indices_obj, num_boxes_sub, num_boxes_obj):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits_sub = outputs['pred_sub_logits']
        pred_logits_obj = outputs['pred_obj_logits']

        device = pred_logits_sub.device

        tgt_lengths_sub = torch.as_tensor([len(v["sub"]) for v in targets], device=device)
        tgt_lengths_obj = torch.as_tensor([len(v["obj"]) for v in targets], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred_sub = (pred_logits_sub.argmax(-1) != pred_logits_sub.shape[-1] - 1).sum(1)
        card_pred_obj = (pred_logits_obj.argmax(-1) != pred_logits_obj.shape[-1] - 1).sum(1)

        card_err_sub = F.l1_loss(card_pred_sub.float(), tgt_lengths_sub.float())
        card_err_obj = F.l1_loss(card_pred_obj.float(), tgt_lengths_obj.float())

        losses = {'cardinality_error_sub': card_err_sub}
        losses = {'cardinality_error_obj': card_err_obj}
        return losses

    '''def loss_relations(self, outputs, targets, indices, num_boxes_sub, num_boxes_obj, log=True):
        """Compute the predicate classification loss
        """
        assert 'pred_rel' in outputs

        src_logits = outputs['pred_rel']
        idx = self._get_src_permutation_idx(indices[2])
        target_classes_o = torch.cat([t["rel"][J] for t, (_, J) in zip(targets, indices[2])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses'''
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss, outputs, targets, indices_sub, indices_obj, num_boxes_sub, num_boxes_obj, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
            #'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices_sub, indices_obj, num_boxes_sub, num_boxes_obj, **kwargs)
    
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_sub = self.matcher_sub(outputs_without_aux, targets)
        indices_obj = self.matcher_obj(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_sub = sum(len(t["sub"]) for t in targets)
        num_boxes_sub = torch.as_tensor([num_boxes_sub], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_obj = sum(len(t["obj"]) for t in targets)
        num_boxes_obj = torch.as_tensor([num_boxes_obj], dtype=torch.float, device=next(iter(outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_sub)
            torch.distributed.all_reduce(num_boxes_obj)
        
        num_boxes_sub = torch.clamp(num_boxes_sub / get_world_size(), min=1).item()
        num_boxes_obj = torch.clamp(num_boxes_obj / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices_sub, indices_obj, num_boxes_sub, num_boxes_obj))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_sub = self.matcher_sub(aux_outputs, targets)
                indices_obj = self.matcher_obj(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_sub, indices_obj, num_boxes_sub, num_boxes_obj, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    

class PostProcessSub(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits_sub, out_bbox_sub = outputs['pred_sub_logits'], outputs['pred_boxes_sub']

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob_sub = F.softmax(out_logits_sub, -1)
        scores_sub, labels_sub = prob_sub[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes_sub = box_cxcywh_to_xyxy(out_bbox_sub)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes_sub = boxes_sub * scale_fct[:, None, :]

        results = [{'scores': s_s, 'labels': l_s, 'boxes': b_s} 
                   for s_s, l_s, b_s in zip(scores_sub, labels_sub, boxes_sub)]

        return results
    
class PostProcessObj(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits_obj, out_bbox_obj = outputs['pred_obj_logits'], outputs['pred_boxes_obj']

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob_obj = F.softmax(out_logits_obj, -1)
        scores_obj, labels_obj = prob_obj[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes_obj = box_cxcywh_to_xyxy(out_bbox_obj)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes_obj = boxes_obj * scale_fct[:, None, :]

        results = [{'scores': s_o, 'labels': l_o, 'boxes': b_o} 
                   for s_o, l_o, b_o in zip(scores_obj, labels_obj, boxes_obj)]

        return results

def build(device, aux_loss, dec_layers):

    hidden_dim = 256
    num_sub = 100
    num_obj=100
    num_queries = 100
    num_classes=181
    num_rel = 51

    backbone = build_backbone()
    transformer = build_transformer(d_model=hidden_dim)

    model = SGG(backbone, 
                transformer=transformer,
                num_queries=num_queries,
                num_classes=num_classes,
                num_rel_classes = num_rel,
                aux_loss=aux_loss
                )
    matcher_sub = build_matcher_sub()
    matcher_obj = build_matcher_obj()

    weight_dict = {
                   'loss_ce_sub':1,
                   'loss_ce_obj': 1,
                   #'loss_bbox': 5,
                   'loss_bbox_sub': 5,
                   'loss_bbox_obj': 5
                   }
    #weight_dict['loss_giou'] = 2
    weight_dict['loss_giou_sub'] = 2
    weight_dict['loss_giou_obj'] = 2
    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, 
                             matcher_sub=matcher_sub, 
                             matcher_obj=matcher_obj, 
                             weight_dict=weight_dict, 
                             num_rel_classes= num_rel,
                             eos_coef=0.1, 
                             losses=losses)
    
    criterion.to(device=device)

    postprocessors_sub = {'bbox': PostProcessSub()}
    postprocessors_obj = {'bbox': PostProcessObj()}
    return model, criterion, postprocessors_sub, postprocessors_obj
