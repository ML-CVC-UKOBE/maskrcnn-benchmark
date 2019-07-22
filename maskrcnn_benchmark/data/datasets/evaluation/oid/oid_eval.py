import pandas as pd
import torch
from tqdm import tqdm
import os

#from object_detection.metrics import io_utils
#from object_detection.metrics import oid_challenge_evaluation_utils as utils
#from object_detection.utils import object_detection_evaluation


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

    # Prepare boxes for detection evaluation
    for i, pred in enumerate(predictions):

        width, height = pred.size
        reescale = torch.tensor(([width, height]*2), dtype=torch.float)[None, ]
        pred.bbox.div_(reescale)
        image_id = dataset.ids[i]

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
            '''
            if image_id not in preds_dict:
                [image_id] = label + " "
            else:
                preds_dict[image_id] += label + " "

            for b in pred.bbox[i]:
                coord = str(b.item()) + " "
                preds_dict[image_id] += coord

        all_predictions = pd.DataFrame.from_dict(
            {"ImageID": preds_dict.keys(),
            "PredictionString": preds_dict.values()})
            '''

    df = {"ImageID": images_id, "LabelName": labels, "Score": scores,
          "XMin": xmin, "YMin": ymin, "XMax": xmax, "YMax": ymax}

    all_predictions = pd.DataFrame.from_dict(df)
    all_location_annotations = dataset.annotations
    all_label_annotations = dataset.image_ann
    all_label_annotations.rename(
        columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)

    is_instance_segmentation_eval = False
    all_annotations = pd.concat(
        [all_location_annotations, all_label_annotations])

    class_label_map = dataset.json_category_id_to_contiguous_id
    categories = [{'id': k, 'name': v}
                  for k, v in dataset.contiguous_category_id_to_json_id.items()]
    challenge_evaluator = (
        object_detection_evaluation.OpenImagesChallengeEvaluator(
            categories, evaluate_masks=is_instance_segmentation_eval))

    images_processed = 0
    for groundtruth in tqdm(all_annotations.groupby('ImageID')):
        image_id, image_groundtruth = groundtruth
        groundtruth_dictionary = utils.build_groundtruth_dictionary(
            image_groundtruth, class_label_map)
        challenge_evaluator.add_single_ground_truth_image_info(
            image_id, groundtruth_dictionary)

        prediction_dictionary = utils.build_predictions_dictionary(
            all_predictions.loc[all_predictions['ImageID'] == image_id],
            class_label_map)
        challenge_evaluator.add_single_detected_image_info(
            image_id, prediction_dictionary)
        images_processed += 1

    metrics = challenge_evaluator.evaluate()
    print(metrics)

    with open(os.path.join(output_folder, "output_metrics.csv"), 'w') as fid:
        io_utils.write_csv(fid, metrics)
