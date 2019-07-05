# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # TODO make this a new config setting
        if 1:
            self.show_boxes(images, proposals, "proposals")
            self.show_boxes(images, result, "detections")

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

    def show_boxes(self, images, proposals, title=""):
        import cv2
        # import numpy as np
        # shape = images.tensors[0].shape
        # image = np.zeros((shape[1], shape[2], shape[0]), dtype='uint8')
        img = images.tensors[0].permute([1, 2, 0]).cpu().numpy()
        if img.max() > 1000:
            img += [123, 116, 102]
            img = img.astype('uint8')
        else:
            img *= [0.229, 0.224, 0.225]
            img += [0.485, 0.456, 0.406]

            img = (img * 255)
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        color = (255, 255, 0)
        max_boxes = 1000
        if isinstance(proposals[0], list):
            list_of_boxes = []
            for b in proposals[0]:
                list_of_boxes.append(b.bbox[:max_boxes, :])
        else:
            list_of_boxes = [proposals[0].bbox[:max_boxes, :]]

        for boxes in list_of_boxes:
            boxes = boxes.to(torch.int64)
            for i, box in enumerate(boxes):
                # if i > max_boxes:
                #     break
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                img = cv2.rectangle(
                    img, tuple(top_left), tuple(bottom_right), tuple(color), 1
                )
        cv2.imshow(title, img)
        cv2.waitKey(0)
