# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from Datasets.Util import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
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
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs_obj, num_queries_obj = outputs["pred_obj_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob_obj = outputs["pred_obj_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox_obj = outputs["pred_boxes_obj"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes

        tgt_ids_obj = torch.cat([v["obj"] for v in targets])
        tgt_bbox_obj = torch.cat([v["objBbox"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class_obj = -out_prob_obj[:, tgt_ids_obj]

        # Compute the L1 cost between boxes
        cost_bbox_obj = torch.cdist(out_bbox_obj, tgt_bbox_obj, p=1)

        # Compute the giou cost betwen boxes
        cost_giou_obj = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_obj), box_cxcywh_to_xyxy(tgt_bbox_obj))

        # Final cost matrix

        C_obj = self.cost_bbox * cost_bbox_obj + \
            self.cost_class * cost_class_obj + \
            self.cost_giou * cost_giou_obj
        C_obj = C_obj.view(bs_obj, num_queries_obj, -1).cpu()

        sizes_obj = [len(v["objBbox"]) for v in targets]
        indices_obj = [linear_sum_assignment(c[i]) for i, c in enumerate(C_obj.split(sizes_obj, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_obj]

def build_matcher_obj():
    return HungarianMatcher(
        cost_class=1, 
        cost_bbox=5, 
        cost_giou=2
    )