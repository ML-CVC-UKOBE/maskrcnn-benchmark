from .oid_eval import do_oid_evaluation, do_oid_submission
from .oid_visualization import do_oid_visualization
from maskrcnn_benchmark.config import cfg


def oid_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if cfg.DATASETS.SUBMIT_ONLY:
        return do_oid_submission(dataset=dataset, predictions=predictions, output_folder=output_folder)

    elif cfg.DATASETS.VISUALIZE:
        return do_oid_visualization(dataset=dataset, predictions=predictions, output_folder=output_folder)

    else:
        return do_oid_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
