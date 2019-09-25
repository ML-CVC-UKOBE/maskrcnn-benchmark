import json

import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np

from .metrics import io_utils
from .metrics import oid_challenge_evaluation_utils as utils
from .utils import object_detection_evaluation


def do_oid_submission(dataset, predictions, output_folder):

    images_id = []
    labels = []
    preds_dict = {}
    missed = []

    dataset.logger.info("")
    dataset.logger.info("Prepare Predictions for Submission")
    dataset.logger.info("")
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):

        image_id = dataset.ids[i]

        # if nothing was detected in the image append dummy prediction
        if len(pred) == 0:
            preds_dict[image_id] = "/m/05s2s 0.0 0.46 0.08 0.93 0.5"
            missed.append(i)

        else:
            width, height = pred.size
            reescale = torch.tensor(([width, height] * 2), dtype=torch.float)[None,]
            pred.bbox.div_(reescale)

            # Extract labels and scores for each box in image
            for i, (label_id, score) in enumerate(
                zip(pred.get_field("labels"), pred.get_field("scores"))
            ):

                label = dataset.contiguous_category_id_to_json_id[label_id.item()]

                labels.append(label)
                images_id.append(image_id)

                if image_id not in preds_dict:
                    preds_dict[image_id] = label + " "
                else:
                    preds_dict[image_id] += label + " "


                preds_dict[image_id] += "{:.8f} ".format(score)
                for b in pred.bbox[i]:
                    coord = "{:.6f} ".format(b.item())
                    preds_dict[image_id] += coord

    all_predictions = pd.DataFrame.from_dict(
        {
            "ImageID": [*preds_dict.keys()],
            "PredictionString": [*preds_dict.values()],
        }
    )

    dataset.logger.info("Predictions Generated. Missed {} images entirely".format(len(missed)))
    dataset.logger.info("Missed Images:")
    dataset.logger.info("\n".join("{}".format(m) for m in missed))
    dataset.logger.info(
        "Saving Submissions To {}".format(
            os.path.join(output_folder, "submission.csv")
        )
    )
    all_predictions.to_csv(
        os.path.join(output_folder, "submission.csv"), index=False
    )

def do_oid_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    images_id = []
    labels = []
    scores = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    dataset.logger.info("")
    dataset.logger.info("Preparing BOXES for evaluation")
    dataset.logger.info("")

    all_predictions_dict = {}
    verbose = False
    expand_leaves_to_hierarchy = True

    if 0:
        show_precomputed_curves(dataset, output_folder)
    # Prepare boxes for detection evaluation
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):
        if i > 100:
            break
        if verbose:
            img, gt, ii = dataset[i]
            show_boxes(
                img,
                [gt],
                "GT",
                dataset.contiguous_category_id_to_json_id,
                dataset.categories,
            )
            # img, gt, ii = dataset[i]
            # show_boxes(img, [pred], "PRED", dataset.contiguous_category_id_to_json_id, dataset.categories)

        boxes = pred.bbox.cpu().numpy()
        boxes /= pred.size * 2
        image_id = dataset.id_to_img_map[i]

        labels = [
            dataset.contiguous_category_id_to_json_id[l.item()]
            for l in pred.get_field("labels")
        ]
        scores = [s.item() for s in pred.get_field("scores")]
        if expand_leaves_to_hierarchy:
            old_labels = labels.copy()
            for il, l in enumerate(old_labels):
                parents = dataset.parents_hierarchy[l]
                n_parents = len(parents)
                if n_parents > 0:
                    labels.extend(list(parents))
                    scores.extend(np.repeat(scores[il], n_parents))
                    boxes = np.vstack(
                        [
                            boxes,
                            np.repeat(boxes[il, :], n_parents)
                            .reshape(4, -1)
                            .transpose(),
                        ]
                    )

                # for new_label in parents:
                #     labels.append(new_label)
                #     scores.append(new_score)
                #     boxes = np.vstack([boxes, new_boxes])
        df = {
            "ImageID": image_id,
            "LabelName": labels,
            "Score": scores,
            "XMin": boxes[:, 0],
            "YMin": boxes[:, 1],
            "XMax": boxes[:, 2],
            "YMax": boxes[:, 3],
        }

        all_predictions_dict[image_id] = pd.DataFrame.from_dict(df)

    all_predictions = pd.concat(all_predictions_dict)
    all_location_annotations = dataset.detections_ann
    all_label_annotations = dataset.image_ann
    all_label_annotations.rename(
        columns={"Confidence": "ConfidenceImageLabel"}, inplace=True
    )

    is_instance_segmentation_eval = False
    all_annotations = pd.concat(
        [all_location_annotations, all_label_annotations], sort=False
    )

    class_label_map = dataset.json_category_id_to_contiguous_id
    categories = [
        {"id": k, "name": v}
        for k, v in dataset.contiguous_category_id_to_json_id.items()
    ]
    challenge_evaluator = object_detection_evaluation.OpenImagesChallengeEvaluator(
        categories, evaluate_masks=is_instance_segmentation_eval
    )

    images_processed = 0
    n_empty = 0

    df = {
        "ImageID": [],
        "LabelName": [],
        "Score": [],
        "XMin": [],
        "YMin": [],
        "XMax": [],
        "YMax": [],
    }
    empty_predictions = pd.DataFrame.from_dict(df)

    dataset.logger.info("")
    dataset.logger.info("Processing EVALUATION")
    dataset.logger.info("")
    all_annotations_grouped = all_annotations.groupby("ImageID")
    for i, groundtruth in enumerate(tqdm(all_annotations_grouped)):
        # if i > 100:
        #     break

        image_id, image_groundtruth = groundtruth
        groundtruth_dictionary = utils.build_groundtruth_dictionary(
            image_groundtruth, class_label_map
        )

        challenge_evaluator.add_single_ground_truth_image_info(
            image_id, groundtruth_dictionary
        )

        if image_id in all_predictions_dict:
            predictions_this_image_id = all_predictions_dict[image_id]
        else:
            predictions_this_image_id = empty_predictions.copy()
            n_empty += 1
        prediction_dictionary = utils.build_predictions_dictionary(
            predictions_this_image_id, class_label_map
        )

        challenge_evaluator.add_single_detected_image_info(
            image_id, prediction_dictionary
        )

        images_processed += 1

    dataset.logger.info("EMPTY images: {}".format(n_empty))
    metrics = challenge_evaluator.evaluate()
    dataset.logger.info(
        metrics["OpenImagesDetectionChallenge_Precision/mAP@0.5IOU"]
    )

    file_output = (
        "results.csv"
        if expand_leaves_to_hierarchy is False
        else "results_expanded.csv"
    )
    with open(os.path.join(output_folder, file_output), "w") as fid:
        io_utils.write_csv(fid, metrics)

    class ComplexEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, complex):
                return [obj.real, obj.imag]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(os.path.join(output_folder, "curves.json"), "w") as fid:
        data = {"precision": challenge_evaluator._evaluation.precisions_per_class,
                "recall": challenge_evaluator._evaluation.recalls_per_class}
        json.dump(data, fid, cls=ComplexEncoder)

    show_precomputed_curves(dataset, output_folder)


def show_precomputed_curves(dataset, output_folder):
    if os.path.exists(os.path.join(output_folder, "curves.json")):
        with open(os.path.join(output_folder, "curves.json"), "r") as fid:
            data = json.load(fid)
        with open(os.path.join(output_folder, "results_expanded.csv"), "r") as fid:
            metrics = pd.read_csv(fid, names=["Category", "AP"])

        metrics["Category"] = metrics["Category"].str.replace("OpenImagesDetectionChallenge_PerformanceByCategory/AP@0.5IOU/", "")
        metrics["Name"] = metrics["Category"].apply(lambda x: dataset.categories[x] if x in dataset.categories else "")
        import pylab
        step_ap = 0.2

        for c_ap in np.arange(0, 1, step_ap):
            for cl in range(500):
                category_id = dataset.contiguous_category_id_to_json_id[cl+1]
                ap = metrics[metrics["Category"] == category_id]["AP"].iloc[0]
                if isinstance(data["precision"][cl], list) and (c_ap <= ap <= c_ap + step_ap):
                    pylab.plot(data["recall"][cl], data["precision"][cl])
            pylab.show()

        counts = dataset.detections_ann.groupby("LabelName")["ImageID"].count().reset_index().rename(columns={"ImageID": "n"})
        step_counts = [0, 10, 50, 100, 1000, 10000000]
        for icount in range(len(step_counts)-1):
            print("----")
            for cl in range(500):
                category_id = dataset.contiguous_category_id_to_json_id[cl+1]
                n_items = counts[counts["LabelName"] == category_id]
                if n_items.shape[0] > 0:
                    n = counts[counts["LabelName"] == category_id]["n"].iloc[0]
                    ap = metrics[metrics["Category"] == category_id]["AP"].iloc[0]
                    if isinstance(data["precision"][cl], list) and (step_counts[icount] <= n <= step_counts[icount+1]):
                        pylab.plot(data["recall"][cl], data["precision"][cl])
                        print("{:<15} {:<30} {:<10} {:<10} {:.3f}".format(category_id,
                                                                          dataset.categories[category_id],
                                                                          n,
                                                                          len(data["recall"][cl]),
                                                                          ap))
            pylab.show()





def show_boxes(images, proposals, title, labels_i_to_id, labels_id_to_name):
    import cv2
    import numpy as np

    # shape = images.tensors[0].shape
    # image = np.zeros((shape[1], shape[2], shape[0]), dtype='uint8')
    img = images.permute([1, 2, 0]).cpu().numpy()
    if img.max() > 10:
        img += [102.9801, 115.9465, 122.7717]
        img = img.astype("uint8")
    else:
        img *= [0.229, 0.224, 0.225]
        img += [0.485, 0.456, 0.406]

        img = img * 255
        img = img.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    max_boxes = 10000
    if isinstance(proposals[0], list):
        list_of_boxes = []
        for b in proposals[0]:
            list_of_boxes.append(b.bbox[:max_boxes, :])
    else:
        list_of_boxes = [proposals[0].bbox[:max_boxes, :]]
        scores = []
        if "objectness" in proposals[0].fields():
            scores = proposals[0].get_field("objectness").cpu().numpy()
        elif "scores" in proposals[0].fields():
            scores = proposals[0].get_field("scores").cpu().numpy()
        else:
            scores = np.ones((len(list_of_boxes), 1))

    labels = proposals[0].get_field("labels")

    for boxes in list_of_boxes:
        boxes = boxes.to(torch.int64)
        for i, box in enumerate(boxes):
            # if i > max_boxes:
            #     break
            if scores[i] == 1:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            img = cv2.rectangle(
                img, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
            label_name = labels_id_to_name[labels_i_to_id[labels[i].item()]]
            text = str(label_name.encode("ascii", "ignore"))
            size, baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            # r = (box[0][1], box[0][0])
            r = top_left[::-1]
            cv2.rectangle(
                img,
                (r[1], r[0]),
                (r[1] + size[0], r[0] + size[1] * 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                img,
                text,
                (r[1], r[0] + baseline * 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    cv2.imshow(title, img)
    cv2.waitKey(0)
