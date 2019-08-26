# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import collections
import json

# import time
import logging
import torch
import torchvision

import numpy as np
import pandas as pd
import tqdm

from PIL import Image
import os
import os.path

from maskrcnn_benchmark.structures.bounding_box import BoxList

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


def convert_hierarchy(hierarchy):
    if isinstance(hierarchy, list):
        list_of_child = []
        for item in hierarchy:
            list_of_child.append(convert_hierarchy(item))
        return list_of_child
    elif isinstance(hierarchy, dict):
        if "Subcategory" in hierarchy:
            list_of_child = convert_hierarchy(hierarchy["Subcategory"])
            return {hierarchy["LabelName"]: list_of_child}
        else:
            return {hierarchy["LabelName"]: []}


def find_leaves(hierarchy):
    if isinstance(hierarchy, list):
        list_of_child = set()
        for item in hierarchy:
            child = find_leaves(item)
            if isinstance(child, set):
                list_of_child = list_of_child.union(child)
            else:
                list_of_child.add(child)
        return list_of_child
    elif isinstance(hierarchy, dict):
        if "Subcategory" in hierarchy:
            return find_leaves(hierarchy["Subcategory"])
        else:
            return hierarchy["LabelName"]


def find_parents(hierarchy, p=set()):
    parents = p.copy()
    if isinstance(hierarchy, list):
        list_of_child = {}
        for item in hierarchy:
            child = find_parents(item, parents)
            # if isinstance(child, list):
            for k in child.keys():
                if k in list_of_child:
                    list_of_child[k] = list_of_child[k].union(child[k])
                else:
                    list_of_child[k] = child[k]
            # list_of_child.update(child)
            # else:
            #     list_of_child.append(child)

        return list_of_child
    elif isinstance(hierarchy, dict):
        if "Subcategory" in hierarchy:
            output = dict()
            if hierarchy["LabelName"] != '/m/0bl9f':  # entity
                output[hierarchy["LabelName"]] = parents.copy()
                parents.add(hierarchy["LabelName"])
            list_of_child = find_parents(hierarchy["Subcategory"], parents)
            output.update(list_of_child)
            return output
        else:
            return {hierarchy["LabelName"]: parents}


class OpenImagesDataset(torch.utils.data.Dataset):
    def __init__(
            self, ann_file, classname_file, hierarchy_file, image_ann_file, images_info_file, root,
            remove_images_without_annotations, filter_subset=(), use_image_labels=False, transforms=None
    ):
        self.fast_init_slow_train = False

        super(OpenImagesDataset, self).__init__()
        self.root = root
        self.logger = logging.getLogger("maskrcnn_benchmark.trainer")
        self.logger.info("Reading OpenImagesDataset Annotations")
        self.detections_ann = pd.read_csv(ann_file)
        self.image_ann = pd.read_csv(image_ann_file)
        self.use_image_labels = use_image_labels

        self.logger.info("Reading OpenImagesDataset Extras")
        self.classname = pd.read_csv(classname_file, header=None, names=["LabelName", "Description"])
        self.categories = collections.OrderedDict()
        for cat in self.classname.values:
            self.categories[cat[0]] = cat[1]

        with open(hierarchy_file) as json_file:
            self.hierarchy = json.load(json_file)
        with open(images_info_file) as json_file:
            self.images_info = json.load(json_file)

        # self.class_hierarchy = convert_hierarchy(self.hierarchy)
        self.leaves_categories = find_leaves(self.hierarchy)
        self.parents_hierarchy = find_parents(self.hierarchy)

        self.logger.info("Filtering Annotations by subset: {}".format(filter_subset))
        for one_filter_subset in filter_subset:
            self.filter_annotations(one_filter_subset)

        if self.fast_init_slow_train:
            self.logger.info("### CAUTION. DEBUGGING PURPOSES --> fast init slow train ###")
            self.ids = self.detections_ann["ImageID"].unique()
        else:
            self.logger.info("Converting Pandas to Dict of Pandas")
            self.annotations = {}
            for groundtruth in tqdm.tqdm(self.detections_ann.groupby('ImageID')):
                image_id, image_groundtruth = groundtruth
                self.annotations[image_id] = image_groundtruth

            self.image_annotations = {}
            for groundtruth in tqdm.tqdm(self.image_ann.groupby('ImageID')):
                image_id, image_groundtruth = groundtruth
                self.image_annotations[image_id] = image_groundtruth

            self.ids = [*self.annotations.keys()]
            # sort indices for reproducible results
        self.ids.sort()
        self.logger.info("Finishing Initialization..")

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.categories)
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.logger.info("Done. {} Images".format(self.__len__()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_id = self.id_to_img_map[idx]

        if self.fast_init_slow_train:
            anno = self.detections_ann[self.detections_ann["ImageID"] == image_id]
            image_anno = self.image_ann[self.image_ann["ImageID"] == image_id]
        else:
            anno = self.annotations[image_id]
            image_anno = self.image_annotations[image_id]

        # HACK to reduce the number of boxes:
        if len(anno) > 500:
            label_type_in_image = anno['LabelName'].unique()
            # leaves_in_image = self.leaves_categories.intersection(label_type_in_image)
            # keep_labels = list()
            labels_to_delete = set()
            for l in label_type_in_image:
                labels_to_delete = labels_to_delete.union(self.parents_hierarchy[l])

            anno = anno[~anno["LabelName"].isin(list(labels_to_delete))]

            if len(anno) > 500:
                anno = anno.sample(500)

        imagename = image_id + ".jpg"
        img = Image.open(os.path.join(self.root, imagename)).convert('RGB')

        # guard against no boxes
        boxes = [anno["XMin"].values, anno["YMin"].values,
                 anno["XMax"].values, anno["YMax"].values]
        boxes = torch.as_tensor(boxes).t().reshape(-1, 4)
        width, height = img.size
        reescale = torch.tensor(([width, height]*2), dtype=torch.float)[None, ]
        target = BoxList(boxes, img.size, mode="xyxy")
        target.bbox.mul_(reescale)

        classes = anno["LabelName"]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if self.use_image_labels:
            list_target_positive = []
            list_target_negative = []
            for c in classes:
                image_classes_positive = image_anno[image_anno["Confidence"] == 1]["LabelName"]
                image_classes_positive = [self.json_category_id_to_contiguous_id[cc] for cc in image_classes_positive]
                if len(image_classes_positive) == 0:
                    image_classes_positive = [0]
                image_classes_positive = torch.tensor(image_classes_positive, dtype=torch.int64)

                image_classes_negative = image_anno[image_anno["Confidence"] == 0]["LabelName"]
                image_classes_negative = [self.json_category_id_to_contiguous_id[cc] for cc in image_classes_negative]
                if len(image_classes_negative) == 0:
                    image_classes_negative = [0]
                image_classes_negative = torch.tensor(image_classes_negative, dtype=torch.int64)

                list_target_positive.append(image_classes_positive)
                list_target_negative.append(image_classes_negative)
            target.add_field("image_labels_positive", torch.stack(list_target_positive))
            target.add_field("image_labels_negative", torch.stack(list_target_negative))

        target = target.clip_to_image(remove_empty=True)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.images_info[img_id]
        return img_data

    def _do_filter_annotations_group_of(self, pd1):
        self.logger.info('Filter by group of [input: {}]'.format(pd1.shape[0]))
        pd1 = pd1[pd1["IsGroupOf"] == 0]
        self.logger.info('Filter by group of [output: {}]'.format(pd1.shape[0]))
        return pd1

    def _do_filter_annotations_leaves(self, pd1):
        self.logger.info('Filter by leaves [input: {}]'.format(pd1.shape[0]))
        pd1 = pd1[pd1["LabelName"].isin(self.leaves_categories)]
        self.logger.info('Filter by leaves [output: {}]'.format(pd1.shape[0]))
        return pd1

    def _do_filter_annotations_group_by_count(self, pd1, pd2, params):
        self.logger.info('Filter by count {} to {} [input: {}]'.format(params[0], params[1], pd1.shape[0]))
        counts = pd1.groupby("LabelName").count().sort_values('ImageID').reset_index()[['LabelName', 'ImageID']]
        labels_to_use = counts[int(params[0]):int(params[1])]['LabelName'].values
        labels_to_use = [str(l) for l in labels_to_use]
        pd1 = pd1[pd1['LabelName'].isin(labels_to_use)]
        pd2 = pd2[pd2['LabelName'].isin(labels_to_use)]

        self.logger.info('Filter by count {} to {} [output: {}]'.format(params[0], params[1], pd1.shape[0]))
        return pd1, pd2

    def _do_filter_annotations_group_by_topic(self, pd1, pd2, params):
        self.logger.info('Filter by topic [#topics:{}. select:{}] [input: {}]'.format(params[0], params[1], pd1.shape[0]))
        pd1["count"] = 1
        counts = pd1.groupby(['ImageID', 'LabelName'])['count'].count()
        cooccurrences = np.zeros((len(self.categories), len(self.categories)))
        temp_label_to_id = {
            v: i for i, v in enumerate(self.categories)
        }
        temp_id_to_label = {
            v: k for k, v in temp_label_to_id.items()
        }
        self.logger.info("-> Creating co-occurrence matrix")
        for image_id, item in tqdm.tqdm(counts.reset_index().groupby('ImageID')):
            labels_in_image = [str(l) for l in item['LabelName'].values]
            for i in labels_in_image:
                for j in labels_in_image:
                    if temp_label_to_id[i] != temp_label_to_id[j]:
                        cooccurrences[temp_label_to_id[i], temp_label_to_id[j]] += 1

        cooccurrences[cooccurrences > 1] = 1

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=int(params[0]), random_state=0).fit(cooccurrences)

        self.logger.info("-> label distribution:")
        for i in range(int(params[0])):
            self.logger.info("      topic:{} -> {} labels".format(i, np.sum(kmeans.labels_ == i)))

        labels_to_use = [str(temp_id_to_label[l]) for l in np.where(kmeans.labels_ == int(params[1]))[0]]

        pd1 = pd1[pd1['LabelName'].isin(labels_to_use)]
        pd2 = pd2[pd2['LabelName'].isin(labels_to_use)]
        self.logger.info('Filter by topic [#topics:{}. select:{}] [output: {}]'.format(params[0], params[1], pd1.shape[0]))
        return pd1, pd2

    def filter_annotations(self, type_filtering):

        if type_filtering == "no_group_of":
            self.detections_ann = self._do_filter_annotations_group_of(self.detections_ann)
        elif type_filtering == "leaves":
            self.detections_ann = self._do_filter_annotations_leaves(self.detections_ann)
            self.image_ann = self._do_filter_annotations_leaves(self.image_ann)
        elif "groupByCount" in type_filtering:
            parameters = type_filtering.split("_")
            self.detections_ann, self.image_ann = self._do_filter_annotations_group_by_count(self.detections_ann,
                                                                                             self.image_ann,
                                                                                             parameters[1:])
        elif "groupByTopic" in type_filtering:
            parameters = type_filtering.split("_")
            self.detections_ann, self.image_ann = self._do_filter_annotations_group_by_topic(self.detections_ann,
                                                                                             self.image_ann,
                                                                                             parameters[1:])
        else:
            raise RuntimeError("UNKNOWN filter annotations")


