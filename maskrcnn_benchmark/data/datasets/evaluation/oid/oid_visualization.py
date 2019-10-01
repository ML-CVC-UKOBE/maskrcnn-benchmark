import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def do_oid_visualization(dataset, predictions, output_folder):

    images_id = []
    labels = []
    scores = []
    models = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    all_predictions_dict = {}

    # Prepare boxes for detection evaluation
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):

        image_id = dataset.ids[i]
        width, height = pred.size
        reescale = torch.tensor(([width, height] * 2), dtype=torch.float)[None,]
        pred.bbox.div_(reescale)

        # Extract predicted box coordinates of every box in image
        for coord in pred.bbox:
            xmin.append(coord[0].item())
            ymin.append(coord[1].item())
            xmax.append(coord[2].item())
            ymax.append(coord[3].item())

        if pred.has_field("models"):
            ensemble_models = pred.get_field("models")
            ensemble = True
        else:
            ensemble_models = torch.zeros(len(pred.get_field("labels")))
            ensemble = False

        # Extract labels and scores for each box in image
        for i, (label_id, score, model) in enumerate(
            zip(
                pred.get_field("labels"),
                pred.get_field("scores"),
                ensemble_models,
            )
        ):

            label = dataset.contiguous_category_id_to_json_id[label_id.item()]
            labels.append(label)
            images_id.append(image_id)
            scores.append(score.item())
            models.append(model.item())

    df = {
        "ImageID": images_id,
        "LabelName": labels,
        "Score": scores,
        "XMin": xmin,
        "YMin": ymin,
        "XMax": xmax,
        "YMax": ymax,
        "Model": models,
    }

    all_predictions = pd.DataFrame.from_dict(df)

    visualize = True
    ap_score = pd.read_csv(
        os.path.join(output_folder, "results_expanded.csv"),
        header=None,
        names=["Label", "mAP"],
    )

    ap_score.drop(index=0, inplace=True)
    ap_score["Label"] = ap_score["Label"].apply(lambda x: x.split("U/")[1])

    gt = dataset.detections_ann

    if ensemble:
        model_map = torch.load(os.path.join(output_folder, "model_map.pth"))
    else:
        model_map = {0: "main"}

    while visualize:

        cl = input("Enter the index of the class you wish to visualize ")
        label = dataset.contiguous_category_id_to_json_id[int(cl)]
        name = dataset.classname[dataset.classname["LabelName"] == label][
            "Description"
        ].iloc[0]

        print("Class: {}".format(name))
        print("Press 'c' to select new class or 'q' to quit")

        to_visualize = all_predictions[all_predictions["LabelName"] == label]
        to_visualize.sort_values(by="Score", ascending=False, inplace=True)

        for _, obj in to_visualize.iterrows():
            ids = obj["ImageID"]
            path = os.path.join(dataset.root, ids + ".jpg")
            image = cv2.imread(path)
            image = image_resize(image, height=800)
            h, w = image.shape[:2]
            mAP = ap_score[ap_score["Label"] == label]["mAP"].iloc[0]

            gt_dets = gt[gt["ImageID"] == ids]
            gt_boxes = gt_dets.values[:, 2:6].astype("float")
            box = [obj["XMin"], obj["XMax"], obj["YMin"], obj["YMax"]]
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.double)
            box = torch.tensor(box, dtype=torch.double)
            closest = F.pairwise_distance(box, gt_boxes).argmin()
            reescale = torch.tensor([w, w, h, h], dtype=torch.double)
            box = box.mul(reescale).int()
            gt_box = gt_boxes[closest].mul(reescale).int()

            cv2.rectangle(
                image,
                (gt_box[0], gt_box[2]),
                (gt_box[1], gt_box[3]),
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                image, (box[0], box[2]), (box[1], box[3]), (255, 0, 0), 2
            )

            # gt_name = ""
            # for i in range(len(gt_dets["LabelName"])):
            #     if dataset.classname[dataset.classname["LabelName"] == gt_dets["LabelName"].iloc[i]].values[0][1] == name:
            #         gt_name = name

            # if not gt_name:
            #     gt_name = dataset.classname[
            #         dataset.classname["LabelName"]
            #         == gt_dets["LabelName"].iloc[closest.item()]
            #     ].values[0][1]

            gt_name = dataset.classname[
                dataset.classname["LabelName"]
                == gt_dets["LabelName"].iloc[closest.item()]
            ].values[0][1]

            cv2.putText(
                image,
                "{}".format(gt_name),
                (gt_box[1] - 40, gt_box[3] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
            )

            cv2.putText(
                image,
                "{:.3f}".format(obj["Score"]),
                (box[0], box[2] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
            )

            cv2.imshow(
                "Class {} Model {}".format(name, model_map[obj["Model"]]), image
            )
            print("mAP: {}".format(mAP))
            key = cv2.waitKey(0)

            if key == ord("c"):
                cv2.destroyAllWindows()
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                visualize = False
                break
