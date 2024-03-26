# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
"""
Modules to compute the matching cost between the predicted triplet and ground truth triplet.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from Datasets.Util import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
import ConfigArgs as args


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, iou_threshold: float = 0.7):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.iou_threshold = iou_threshold
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]
        #num_queries_rel = outputs["rel_logits"].shape[1]
        alpha = 0.25
        gamma = 2.0

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the entity classification cost. We borrow the cost function from Deformable DETR (https://arxiv.org/abs/2010.04159)
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between entity boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen entity boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final entity cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Concat the subject/object/predicate labels and subject/object boxes
        sub_tgt_bbox = torch.cat([v['boxes'][: int(len(v['boxes']) / 2)] for v in targets])
        sub_tgt_ids = torch.cat([v['labels'][: int(len(v['boxes']) / 2)] for v in targets])
        obj_tgt_bbox = torch.cat([v['boxes'][int(len(v['boxes']) / 2) : ] for v in targets])
        obj_tgt_ids = torch.cat([v['labels'][int(len(v['boxes']) / 2) : ] for v in targets])

        sub_prob = outputs["sub_logits"].flatten(0, 1).sigmoid()
        sub_bbox = outputs["sub_boxes"].flatten(0, 1)
        obj_prob = outputs["obj_logits"].flatten(0, 1).sigmoid()
        obj_bbox = outputs["obj_boxes"].flatten(0, 1)

        # Compute the subject matching cost based on class and box.
        neg_cost_class_sub = (1 - alpha) * (sub_prob ** gamma) * (-(1 - sub_prob + 1e-8).log())
        pos_cost_class_sub = alpha * ((1 - sub_prob) ** gamma) * (-(sub_prob + 1e-8).log())
        cost_sub_class = pos_cost_class_sub[:, sub_tgt_ids] - neg_cost_class_sub[:, sub_tgt_ids]
        cost_sub_bbox = torch.cdist(sub_bbox, sub_tgt_bbox, p=1)
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(sub_tgt_bbox))

        # Compute the object matching cost based on class and box.
        neg_cost_class_obj = (1 - alpha) * (obj_prob ** gamma) * (-(1 - obj_prob + 1e-8).log())
        pos_cost_class_obj = alpha * ((1 - obj_prob) ** gamma) * (-(obj_prob + 1e-8).log())
        cost_obj_class = pos_cost_class_obj[:, obj_tgt_ids] - neg_cost_class_obj[:, obj_tgt_ids]
        cost_obj_bbox = torch.cdist(obj_bbox, obj_tgt_bbox, p=1)
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(obj_tgt_bbox))

        # Final couple cost matrix
        C_rel = self.cost_bbox * cost_sub_bbox + self.cost_bbox * cost_obj_bbox  + \
                self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + \
                self.cost_giou * cost_sub_giou + self.cost_giou * cost_obj_giou
        C_rel = C_rel.view(bs, num_queries, -1).cpu()

        sizes1 = [int(len(v["boxes"])) for v in targets]
        indices1 = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rel.split(sizes1, -1))]

        # assignment strategy to avoid assigning <background-no_relationship-background > to some good predictions
        sub_weight = torch.ones((bs, num_queries)).to(out_prob.device)
        good_sub_detection = torch.logical_and((outputs["sub_logits"].flatten(0, 1)[:, :-1].argmax(-1)[:, None] == tgt_ids),
                                               (box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0] >= self.iou_threshold))
        for i, c in enumerate(good_sub_detection.split(sizes, -1)):
            sub_weight[i, c.sum(-1)[i*num_queries:(i+1)*num_queries].to(torch.bool)] = 0
            sub_weight[i, indices[i][0]] = 1

        obj_weight = torch.ones((bs, num_queries)).to(out_prob.device)
        good_obj_detection = torch.logical_and((outputs["obj_logits"].flatten(0, 1)[:, :-1].argmax(-1)[:, None] == tgt_ids),
                                               (box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0] >= self.iou_threshold))
        for i, c in enumerate(good_obj_detection.split(sizes, -1)):
            obj_weight[i, c.sum(-1)[i*num_queries:(i+1)*num_queries].to(torch.bool)] = 0
            obj_weight[i, indices[i][0]] = 1

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],\
               [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices1],\
               sub_weight, obj_weight


def build_matcher():
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, iou_threshold=args.set_iou_threshold)
