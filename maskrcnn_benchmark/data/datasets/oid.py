# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json

import torch
import torchvision

import pandas as pd

from PIL import Image
import os
import os.path

from maskrcnn_benchmark.structures.bounding_box import BoxList


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    # if "keypoints" not in anno[0]:
    #     return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    # if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    #     return True
    return False


class OpenImagesDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self, ann_file, classname_file, hierarchy_file, image_ann_file, images_info_file, root,
            remove_images_without_annotations, transforms=None
    ):
        super(OpenImagesDataset, self).__init__(root)
        # sort indices for reproducible results

        self.annotations = pd.read_csv(ann_file)
        self.classname = pd.read_csv(classname_file, header=None, names=["LabelName", "Description"])
        self.image_ann = pd.read_csv(image_ann_file)
        with open(hierarchy_file) as json_file:
            self.hierarchy = json.load(json_file)

        with open(images_info_file) as json_file:
            self.images_info = json.load(json_file)

        self.ids = self.annotations["ImageID"].unique()

        self.categories = {cat[0]: cat[1] for cat in self.classname.values}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.categories)
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_id = self.id_to_img_map[idx]
        anno = self.annotations[self.annotations["ImageID"] == image_id]

        imagename = image_id + ".jpg"
        img = Image.open(os.path.join(self.root, imagename)).convert('RGB')

        # filter crowd annotations
        # anno = anno[anno["IsGroupOf"] ==  0]
        boxes = [anno["XMin"].values, anno["YMin"].values,
                 anno["XMax"].values, anno["YMax"].values]
        boxes = torch.as_tensor(boxes).t().reshape(-1, 4)  # guard against no boxes
        boxes[:, 0] *= img.size[0]
        boxes[:, 2] *= img.size[0]
        boxes[:, 1] *= img.size[1]
        boxes[:, 3] *= img.size[1]
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = anno["LabelName"]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # if anno and "segmentation" in anno[0]:
        #     masks = [obj["segmentation"] for obj in anno]
        #     masks = SegmentationMask(masks, img.size, mode='poly')
        #     target.add_field("masks", masks)
        #
        # if anno and "keypoints" in anno[0]:
        #     keypoints = [obj["keypoints"] for obj in anno]
        #     keypoints = PersonKeypoints(keypoints, img.size)
        #     target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.images_info[img_id]
        return img_data
