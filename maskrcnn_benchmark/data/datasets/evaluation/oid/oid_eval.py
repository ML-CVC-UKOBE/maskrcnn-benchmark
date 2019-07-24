import pandas as pd
import torch
from tqdm import tqdm
import os

from .metrics import io_utils
from .metrics import oid_challenge_evaluation_utils as utils
from .utils import object_detection_evaluation


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
    print("Preparing BOXES for evaluation")
    verbose = False
    # Prepare boxes for detection evaluation
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):

        if verbose:
            img, gt, ii = dataset[i]
            show_boxes(img, [gt], "GT", dataset.contiguous_category_id_to_json_id, dataset.categories)
            img, gt, ii = dataset[i]
            show_boxes(img, [pred], "PRED", dataset.contiguous_category_id_to_json_id, dataset.categories)

        width, height = pred.size
        reescale = torch.tensor(([width, height]*2), dtype=torch.float)[None, ]
        pred.bbox.div_(reescale)
        # image_id2 = dataset.ids[i]
        image_id = dataset.id_to_img_map[i]
        # Extract predicted box coordinates of every box in image
        for coord in pred.bbox:
            xmin.append(coord[0].item())
            ymin.append(coord[1].item())
            xmax.append(coord[2].item())
            ymax.append(coord[3].item())

        # Extract labels and scores for each box in image
        for j, (label_id, score) in enumerate(zip(pred.get_field('labels'),
                                                  pred.get_field('scores'))):

            label = dataset.contiguous_category_id_to_json_id[label_id.item()]

            labels.append(label)
            images_id.append(image_id)
            scores.append(score.item())

    df = {"ImageID": images_id, "LabelName": labels, "Score": scores,
          "XMin": xmin, "YMin": ymin, "XMax": xmax, "YMax": ymax}

    all_predictions = pd.DataFrame.from_dict(df)

    print("Converting Predictions to Dict")
    all_predictions_grouped = all_predictions.groupby('ImageID')
    all_predictions_dict = {}
    for prediction in tqdm(all_predictions_grouped):
        image_id, image_prediction = prediction
        all_predictions_dict[image_id] = image_prediction.reset_index(drop=True)

    all_location_annotations = dataset.pd_ann
    all_label_annotations = dataset.image_ann
    all_label_annotations.rename(columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)

    is_instance_segmentation_eval = False
    all_annotations = pd.concat([all_location_annotations, all_label_annotations])

    class_label_map = dataset.json_category_id_to_contiguous_id
    categories = [{'id': k, 'name': v} for k, v in dataset.contiguous_category_id_to_json_id.items()]
    challenge_evaluator = (
        object_detection_evaluation.OpenImagesChallengeEvaluator(
            categories, evaluate_masks=is_instance_segmentation_eval))

    images_processed = 0
    all_annotations_grouped = all_annotations.groupby('ImageID')

    df = {"ImageID": [], "LabelName": [], "Score": [], "XMin": [], "YMin": [], "XMax": [], "YMax": []}
    empty_predictions = pd.DataFrame.from_dict(df)
    n_empty = 0

    for groundtruth in tqdm(all_annotations_grouped):

        image_id, image_groundtruth = groundtruth
        groundtruth_dictionary = utils.build_groundtruth_dictionary(image_groundtruth, class_label_map)

        challenge_evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dictionary)

        if image_id in all_predictions_dict:
            predictions_this_image_id = all_predictions_dict[image_id]
        else:
            predictions_this_image_id = empty_predictions.copy()
            n_empty += 1
        prediction_dictionary = utils.build_predictions_dictionary(predictions_this_image_id, class_label_map)

        challenge_evaluator.add_single_detected_image_info(image_id, prediction_dictionary)

        images_processed += 1

    print("EMPTY images: {}".format(n_empty))
    metrics = challenge_evaluator.evaluate()
    print(metrics["OpenImagesDetectionChallenge_Precision/mAP@0.5IOU"])

    with open(os.path.join(output_folder, "results.csv"), 'w') as fid:
        io_utils.write_csv(fid, metrics)


def show_boxes(images, proposals, title, labels_i_to_id, labels_id_to_name):
    import cv2
    import numpy as np
    # shape = images.tensors[0].shape
    # image = np.zeros((shape[1], shape[2], shape[0]), dtype='uint8')
    img = images.permute([1, 2, 0]).cpu().numpy()
    if img.max() > 10:
        img += [102.9801, 115.9465, 122.7717]
        img = img.astype('uint8')
    else:
        img *= [0.229, 0.224, 0.225]
        img += [0.485, 0.456, 0.406]

        img = (img * 255)
        img = img.astype('uint8')
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
            text = str(label_name.encode('ascii', 'ignore'))
            size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # r = (box[0][1], box[0][0])
            r = top_left[::-1]
            cv2.rectangle(img, (r[1], r[0]), (r[1] + size[0], r[0] + size[1] * 2), (0, 0, 0), -1)
            cv2.putText(img, text, (r[1], r[0] + baseline * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(title, img)
    cv2.waitKey(10)
