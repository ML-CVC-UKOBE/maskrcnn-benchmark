# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import collections
import json
import time

import torch
import torchvision

import numpy as np
import pandas as pd
import tqdm

from PIL import Image
import os
import os.path

from maskrcnn_benchmark.structures.bounding_box import BoxList

pd.set_option('display.max_rows', 20)
# pd.set_option('display.max_columns', 500)
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
            if hierarchy["LabelName"] != '/m/0bl9f':  # entity
                parents.add(hierarchy["LabelName"])
            list_of_child = find_parents(hierarchy["Subcategory"], parents)
            return list_of_child
        else:
            return {hierarchy["LabelName"]: parents}


class OpenImagesDataset(torchvision.datasets.VisionDataset):
    def __init__(
            self, ann_file, classname_file, hierarchy_file, image_ann_file, images_info_file, root,
            remove_images_without_annotations, filter_subset=(), transforms=None
    ):
        super(OpenImagesDataset, self).__init__(root)

        print("Reading OpenImagesDataset Annotations")
        self.detections_ann = pd.read_csv(ann_file)
        self.image_ann = pd.read_csv(image_ann_file)

        print("Reading OpenImagesDataset Extras")
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

        print("Filtering Annotations by subset: {}".format(filter_subset))
        for one_filter_subset in filter_subset:
            self.filter_annotations(one_filter_subset)

        print("Converting Pandas to Dict of Pandas")
        self.annotations = {}
        for groundtruth in tqdm.tqdm(self.detections_ann.groupby('ImageID')):
            image_id, image_groundtruth = groundtruth
            self.annotations[image_id] = image_groundtruth

        self.image_annotations = {}
        for groundtruth in tqdm.tqdm(self.image_ann.groupby('ImageID')):
            image_id, image_groundtruth = groundtruth
            self.image_annotations[image_id] = image_groundtruth

        print("Finishing Initialization..")
        self.ids = [*self.annotations.keys()]
        # sort indices for reproducible results
        self.ids.sort()

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.categories)
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        print("Done. {} Images".format(self.__len__()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_id = self.id_to_img_map[idx]
        anno = self.annotations[image_id]

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

        target = target.clip_to_image(remove_empty=True)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.images_info[img_id]
        return img_data

    def _do_filter_annotations_group_of(self, pd1):
        print('Filter by group of [input: {}]'.format(pd1.shape[0]))
        pd1 = pd1[pd1["IsGroupOf"] == 0]
        print('Filter by group of [output: {}]'.format(pd1.shape[0]))
        return pd1

    def _do_filter_annotations_leaves(self, pd1):
        print('Filter by leaves [input: {}]'.format(pd1.shape[0]))
        pd1 = pd1[pd1["LabelName"].isin(self.leaves_categories)]
        print('Filter by leaves [output: {}]'.format(pd1.shape[0]))
        return pd1

    def _do_filter_annotations_group_by_count(self, pd1, pd2, params):
        print('Filter by count {} to {} [input: {}]'.format(params[0], params[1], pd1.shape[0]))
        counts = pd1.groupby("LabelName").count().sort_values('ImageID').reset_index()[['LabelName', 'ImageID']]
        labels_to_use = counts[int(params[0]):int(params[1])]['LabelName'].values
        labels_to_use = [str(l) for l in labels_to_use]
        pd1 = pd1[pd1['LabelName'].isin(labels_to_use)]
        pd2 = pd2[pd2['LabelName'].isin(labels_to_use)]

        print('Filter by count {} to {} [output: {}]'.format(params[0], params[1], pd1.shape[0]))
        return pd1, pd2

    def _do_filter_annotations_group_by_topic(self, pd1, pd2, params):
        print('Filter by topic [#topics:{}. select:{}] [input: {}]'.format(params[0], params[1], pd1.shape[0]))
        pd1["count"] = 1
        counts = pd1.groupby(['ImageID', 'LabelName'])['count'].count()
        cooccurrences = np.zeros((len(self.categories), len(self.categories)))
        temp_label_to_id = {
            v: i for i, v in enumerate(self.categories)
        }
        temp_id_to_label = {
            v: k for k, v in temp_label_to_id.items()
        }
        print("-> Creating co-occurrence matrix")
        for image_id, item in tqdm.tqdm(counts.reset_index().groupby('ImageID')):
            labels_in_image = [str(l) for l in item['LabelName'].values]
            for i in labels_in_image:
                for j in labels_in_image:
                    if temp_label_to_id[i] != temp_label_to_id[j]:
                        cooccurrences[temp_label_to_id[i], temp_label_to_id[j]] += 1

        cooccurrences[cooccurrences > 1] = 1

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=int(params[0]), random_state=0).fit(cooccurrences)

        print("-> label distribution:")
        for i in range(int(params[0])):
            print("      topic:{} -> {} labels".format(i, np.sum(kmeans.labels_ == i)))

        labels_to_use = [str(temp_id_to_label[l]) for l in np.where(kmeans.labels_ == int(params[1]))[0]]

        pd1 = pd1[pd1['LabelName'].isin(labels_to_use)]
        pd2 = pd2[pd2['LabelName'].isin(labels_to_use)]
        print('Filter by topic [#topics:{}. select:{}] [output: {}]'.format(params[0], params[1], pd1.shape[0]))
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


