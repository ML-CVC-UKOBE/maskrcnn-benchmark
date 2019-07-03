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
        self, ann_file, classname_file, hierarchy_file, image_ann_file, info_images, root,
            remove_images_without_annotations, transforms=None
    ):
        super(OpenImagesDataset, self).__init__(root)
        # sort indices for reproducible results

        self.annotations = pd.read_csv(ann_file)
        self.classname_file = pd.read_csv(classname_file, header=None, names=["LabelName", "Description"])
        self.image_ann_file = pd.read_csv(image_ann_file)
        with open(hierarchy_file) as json_file:
            self.hierarchy_file = json.load(json_file)

        with open(info_images) as json_file:
            self.info_images = json.load(json_file)

        self.annotations = self.annotations.drop(columns=['Source', 'Confidence'])
        self.image_ann_file = self.image_ann_file.drop(columns=['Source'])

        self.ids = self.annotations["ImageID"].unique()

        self.annotations = self.annotations.set_index("ImageID")

        # filter images without detection annotations
        # if remove_images_without_annotations:
        #     ids = []
        #     for img_id in self.ids:
        #         ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #         anno = self.coco.loadAnns(ann_ids)
        #         if has_valid_annotation(anno):
        #             ids.append(img_id)
        #     self.ids = ids

        self.categories = {cat[0]: cat[1] for cat in self.classname_file.values}

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
        # img, anno = super(COCODataset, self).__getitem__(idx)

        image_id = self.id_to_img_map[idx]

        anno = self.annotations[self.annotations["ImageID"] == image_id]

        imagename = image_id + ".jpg"
        img = Image.open(os.path.join(self.root, imagename)).convert('RGB')


        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
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
        img_data = self.coco.imgs[img_id]
        return img_data
