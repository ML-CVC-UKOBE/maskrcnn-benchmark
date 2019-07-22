# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import time

import torch
import torchvision

import re
import pandas as pd
import tqdm

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


class OidRecord:
    def __init__(self, row):
        self.image_id = row[0]
        if len(row) == 13:
            self.is_train = True
        elif len(row) != 7:
            raise RuntimeError("ROW SIZE is not 13 (for training) neither 7 (validation)")

        if self.is_train:
            self.label = row[2]
            self.box = [float(i) for i in row[4:8]]
            # self.confidence = float(row[3])
            self.is_group_of = row[10] == '1'
        else:
            self.label = row[1]
            self.box = [float(i) for i in row[2:6]]
            # self.confidence = 1
            self.is_group_of = row[6] == '1'


class OpenImagesDataset(torchvision.datasets.VisionDataset):
    def __init__(
            self, ann_file, classname_file, hierarchy_file, image_ann_file, images_info_file, root,
            remove_images_without_annotations, transforms=None
    ):
        super(OpenImagesDataset, self).__init__(root)
        # sort indices for reproducible results

        print("Reading OpenImagesDataset Annotations")
        self.classname = pd.read_csv(classname_file, header=None, names=["LabelName", "Description"])
        self.image_ann = pd.read_csv(image_ann_file)
        with open(hierarchy_file) as json_file:
            self.hierarchy = json.load(json_file)

        with open(images_info_file) as json_file:
            self.images_info = json.load(json_file)

        self.annotations = {}

        with open(ann_file, mode='r') as csv_file:
            next(csv_file)  # skip header
            for line in tqdm.tqdm(csv_file.readlines()):
                item = OidRecord(re.split(',', line.replace("\n", "")))
                image_id = item.image_id
                if image_id not in self.annotations:
                    self.annotations[image_id] = []
                self.annotations[image_id].append(item)

        self.ids = self.annotations.keys()

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
        anno = self.annotations[image_id]

        imagename = image_id + ".jpg"
        img = Image.open(os.path.join(self.root, imagename)).convert('RGB')

        boxes = []
        classes = []
        for i, a in enumerate(anno):
            # if a.is_group_of:
                # filter crowd annotations
                # continue
            boxes.append((a.box[0], a.box[2], a.box[1], a.box[3]))  # xmin(), ymin(), xmax(), ymax()
            classes.append(a.label)
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

        width, height = img.size
        reescale = torch.tensor(([width, height]*2),dtype=torch.float)[None,]
        target = BoxList(boxes, img.size, mode="xyxy")
        target.bbox.mul_(reescale)

        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.images_info[img_id]
        return img_data
