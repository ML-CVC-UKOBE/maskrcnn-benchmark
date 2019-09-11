import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image

def do_oid_visualization(dataset, predictions, output_folder):

    images_id = []
    labels = []
    scores = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    all_predictions_dict = {}

    # Prepare boxes for detection evaluation
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):

        image_id = dataset.ids[i]
        width, height = pred.size
        reescale = torch.tensor(([width, height]*2), dtype=torch.float)[None, ]
        pred.bbox.div_(reescale)

        # Extract predicted box coordinates of every box in image
        for coord in pred.bbox:
            xmin.append(coord[0].item())
            ymin.append(coord[1].item())
            xmax.append(coord[2].item())
            ymax.append(coord[3].item())

        # Extract labels and scores for each box in image
        for i, (label_id, score) in enumerate(zip(pred.get_field('labels'),
                                                  pred.get_field('scores'))):

            label = dataset.contiguous_category_id_to_json_id[label_id.item()]
            labels.append(label)
            images_id.append(image_id)
            scores.append(score.item())


    df = {"ImageID": images_id, "LabelName": labels, "Score": scores,
          "XMin": xmin, "YMin": ymin, "XMax": xmax, "YMax": ymax}

    all_predictions = pd.DataFrame.from_dict(df)

    visualize = True
    ap_score = pd.read_csv(os.path.join(output_folder, "results_expanded.csv"),
                           header=None,names=["Label", "mAP"])

    ap_score.drop(index=0, inplace=True)
    ap_score["Label"] = ap_score["Label"].apply(lambda x: x.split('U/')[1])

    while visualize:

        cl = input("Enter the index of the class you wish to visualize ")
        label = dataset.contiguous_category_id_to_json_id[int(cl)]
        name = dataset.classname[dataset.classname["LabelName"] == label]["Description"].iloc[0]

        print("Class: {}".format(name))
        print("Press 'c' to select new class or 'q' to quit")

        to_visualize = all_predictions[all_predictions["LabelName"] == label]
        to_visualize.sort_values(by="Score", ascending=False, inplace=True)

        for ids in to_visualize["ImageID"].unique():
            path = os.path.join(dataset.root, ids + ".jpg")
            image = cv2.imread(path)
            h, w = image.shape[:2]
            mAP = ap_score[ap_score["Label"] == label]["mAP"].iloc[0]

            obj = to_visualize[to_visualize["ImageID"] == ids].iloc[0]
            box = [obj["XMin"] * w, obj["YMin"] * h, obj["XMax"] * w, obj["YMax"] * h]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)

            cv2.putText(image, str(obj["Score"]), (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

            cv2.imshow("Class {} mAP: {}".format(name, mAP), image)
            key = cv2.waitKey(0)

            if key == ord('c'):
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                visualize = False
                break
