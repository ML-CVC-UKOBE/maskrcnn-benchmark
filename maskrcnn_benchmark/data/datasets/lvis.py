import os
import torch
import torchvision

import pandas as pd

from lvis import LVIS
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


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
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class LvisDataset(LVIS):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(LvisDataset, self).__init__(ann_file)
        # sort indices for reproducible results
        self.root = root
        self.ids = sorted(self.get_img_ids())

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.get_ann_ids(img_ids=[img_id])
                anno = self.load_anns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {k: info["name"] for k, info in self.cats.items()}

        self.json_category_id_to_contiguous_id = {
            v: k for k, v in self.categories.items()
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):

        id_img = self.ids[idx]
        ann_ids = self.get_ann_ids(img_ids=[id_img])
        anno = self.load_anns(ann_ids)

        # HACK to reduce the number of boxes:
        if len(anno) > 400:
            anno = pd.DataFrame(anno)
            #TODO maybe sample from the classes that are already learned
            anno = anno.sample(400)
            anno = anno.to_dict('records')

        img = Image.open(
            os.path.join(self.root, self.imgs[id_img]["file_name"])
        ).convert("RGB")

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode="poly")
            target.add_field("masks", masks)


        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.imgs[img_id]
        return img_data
