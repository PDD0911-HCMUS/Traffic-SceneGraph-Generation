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
        bs_sub, num_queries_sub = outputs["pred_sub_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob_sub = outputs["pred_sub_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        #out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids_sub = torch.cat([v["sub"] for v in targets])
        tgt_bbox_sub = torch.cat([v["subBbox"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class_sub = -out_prob_sub[:, tgt_ids_sub]

        # Compute the L1 cost between boxes
        out_bbox_sub = outputs["pred_boxes_sub"].flatten(0, 1)  # [batch_size * num_queries, 4]
        cost_bbox_sub = torch.cdist(out_bbox_sub, tgt_bbox_sub, p=1)

        # Compute the giou cost betwen boxes
        cost_giou_sub = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_sub), box_cxcywh_to_xyxy(tgt_bbox_sub))

        # Final cost matrix
        C_sub = self.cost_bbox * cost_bbox_sub + \
            self.cost_class * cost_class_sub + \
            self.cost_giou * cost_giou_sub 
        C_sub = C_sub.view(bs_sub, num_queries_sub, -1).cpu()

        sizes_sub = [len(v["subBbox"]) for v in targets]
        indices_sub = [linear_sum_assignment(c[i]) for i, c in enumerate(C_sub.split(sizes_sub, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_sub]
    
def build_matcher_sub():
    return HungarianMatcher(
        cost_class=1, 
        cost_bbox=5, 
        cost_giou=2
    )