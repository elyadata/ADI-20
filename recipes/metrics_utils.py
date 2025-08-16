import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from speechbrain.utils.distributed import main_process_only

logger = logging.getLogger(__name__)


@main_process_only
def make_dir(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def _group_predictions(predictions: list[dict]):
    """
    Group the list of prediction dictionaries into a single dictionary.

    Parameters:
    - predictions (list): A list of prediction dictionaries.

    Returns:
    - dict: A dictionary with keys from the prediction dicts and values as lists of data.
    """
    grouped_dict = {
        "scores": [],
        "indexes": [],
        "predicted_labels": [],
        "target_labels": []
    }

    for prediction in predictions:
        for key in grouped_dict.keys():
            grouped_dict[key].extend(prediction[key])

    return grouped_dict


@main_process_only
def save_predictions_to_json(filename: str, metrics_log_dir: str, predictions: dict):
    """
    Save grouped predictions to a JSON file.

    Parameters:
    - filename (str): The name of the JSON file (default is 'classifications_test.json').
    - metrics_log_dir (str): The directory where the JSON file will be saved.
    - predictions (list): A list of prediction dictionaries to be saved.
    """
    grouped_predictions = _group_predictions(predictions)

    file_path = os.path.join(metrics_log_dir, filename)

    with open(file_path, mode="w") as predictions_file:
        json.dump(grouped_predictions, predictions_file)
    logger.info(f"Predictions saved in {file_path}.")


def _load_predictions_from_json(file_path):
    """
    Load predictions from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: A dictionary containing the loaded predictions.
    """
    with open(file_path, mode="r") as predictions_file:
        predictions = json.load(predictions_file)
    return predictions


def _calculate_classification_metrics(predictions, plot_matrix: bool = False, plot_path: str | None = None, fewer_eval_classes: bool = False):
    """
    Calculate classification metrics from the grouped predictions.

    Parameters:
    - predictions (dict): A dictionary containing grouped predictions.
    - plot_matrix (bool): Whether to plot the confusion matrix.
    - plot_path (str | None): Path to save the confusion matrix plot.

    Returns:
    - tuple: A formatted classification report and the confusion matrix.
    """
    logger.info("Calculating classification metrics and confusion matrics...")
    if isinstance(predictions, list):
        predictions = _group_predictions(predictions)

    true_labels = predictions["target_labels"]
    predicted_labels = predictions["predicted_labels"]

    if not fewer_eval_classes:
        class_labels = sorted(set(true_labels) | set(predicted_labels))
    else:
        class_labels = sorted(set(true_labels))
        unexpected_labels = set(predicted_labels) - set(true_labels)
        if unexpected_labels:
            logger.warning(
                f"The following predicted labels are not in the true labels: {unexpected_labels}")

    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        labels=class_labels,
        zero_division=0.0)
    conf_matrix = confusion_matrix(
        true_labels, predicted_labels, labels=class_labels)
    if plot_matrix:
        if plot_path is None:
            logger.warning(
                "Provide a `plot_path` in order to save the confusion matrix.")
        else:
            logger.info("Generating confusion matrix plot...")
            plt.figure(figsize=(12, 10))
            vmin, vmax = 0, np.max(conf_matrix)
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                vmin=vmin, vmax=vmax,
                annot_kws={"size": 8}
            )
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Confusion matrix plot saved in {plot_path}")

    return report, conf_matrix


@main_process_only
def generate_metrics_report(predictions,
                            plot_matrix: bool = False,
                            plot_path: str | None = None,
                            report_path: str = None,
                            verbose: bool = True,
                            return_dict: bool = False,
                            fewer_eval_classes: bool = False) -> dict | None:
    """
    Generate a classification report and confusion matrix from predictions.

    Parameters:
    - predictions (dict): A dictionary containing grouped predictions.
    - plot_matrix (bool): Whether to plot the confusion matrix.
    - plot_path (str | None): Path to save the confusion matrix plot.

    Returns:
        None
    """

    if report_path is None and not verbose:
        logger.warning(
            "Either specify `verbose=True` or provide a `report_path`.")
        return

    metrics_report, conf_matrix = _calculate_classification_metrics(
        predictions=predictions, plot_matrix=plot_matrix, plot_path=plot_path, fewer_eval_classes=fewer_eval_classes)

    if verbose:
        logger.info(f"Classification Report: \n{metrics_report}")
        logger.info(f"Confusion Matrix: \n{conf_matrix}")

    if report_path:
        logger.info("Generating report...")
        with open(report_path, "w") as report_file:
            json.dump(metrics_report, report_file, indent=4)
        logger.info(f"Report saved in {report_path}")

    if return_dict:
        return {
            "macro_f1": metrics_report["macro avg"]["f1-score"],
            "macro_precision": metrics_report["macro avg"]["precision"],
            "macro_recall": metrics_report["macro avg"]["recall"],
            "weighted_f1": metrics_report["weighted avg"]["f1-score"],
            "weighted_precision": metrics_report["weighted avg"]["precision"],
            "weighted_recall": metrics_report["weighted avg"]["recall"],
        }


@main_process_only
def generate_report_from_file(file_path: str = "classifications_test.json",
                              report_path: str = None,
                              verbose=True,
                              plot_matrix: bool = False,
                              plot_path: str | None = None,
                              fewer_eval_classes: bool = False):
    """
    Main function to load predictions and calculate metrics.

    Parameters:
    - file_path (str): The path to the JSON file containing predictions.
    - report_path (str | None): The path to save the classification report.
    - verbose (bool): Whether to log the output.
    - plot_matrix (bool): Whether to plot the confusion matrix.
    - plot_path (str | None): Path to save the confusion matrix plot.
    """
    generate_metrics_report(predictions=_load_predictions_from_json(file_path),
                            plot_matrix=plot_matrix,
                            plot_path=plot_path,
                            report_path=report_path,
                            verbose=verbose,
                            fewer_eval_classes=fewer_eval_classes)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Generate classification reports for ADI models."
    )

    parser.add_argument("predictions_file", type=str,
                        help="Path to the classification report JSON file.")
    parser.add_argument("--report_path", type=str, default=None,
                        help="Path to save the corrected JSON file.")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose logging (default: True). Use --no-verbose to disable.")
    parser.add_argument("--no-verbose", action="store_false", dest="verbose",
                        help="Disable verbose logging.")
    parser.add_argument("--plot-matrix", action="store_true", default=False,
                        help="Generate and save a confusion matrix plot.")
    parser.add_argument("--plot-path", type=str, default=None,
                        help="Path to save the confusion matrix plot (default: None).")
    parser.add_argument("--fewer_eval_classes", action="store_true", default=False,
                        help="Whether the evaluations set contains fewer classes than the model's train set.")
    args = parser.parse_args()

    logger.info(f"Loading predictions from: {args.predictions_file}")
    logger.info(f"Output file: {args.report_path}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info(f"Plot confusion matrix: {args.plot_matrix}")
    if args.plot_matrix:
        logger.info(
            f"Confusion matrix plot will be saved at: {args.plot_path}")
        
    if args.report_path is None:
        predictions_dir = os.path.dirname(args.predictions_file)
        report_path = os.path.join(predictions_dir, "classification_report_corrected.json")

    generate_report_from_file(
        file_path=args.predictions_file,
        report_path=report_path,
        verbose=args.verbose,
        plot_matrix=args.plot_matrix,
        plot_path=args.plot_path,
        fewer_eval_classes=args.fewer_eval_classes,
    )

    logger.info("Classification report generation completed.")
