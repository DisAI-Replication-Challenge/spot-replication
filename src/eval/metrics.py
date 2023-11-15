import sklearn
import numpy as np
import scipy.stats


def sklearn_metrics_wrapper(metric_str, metric_dict_str=None, metric_post_process_fn=None, **metric_fn_kwargs):
    """Wraps any sklearn.metric function and returns a t5 metric function.

    Args:
      metric_str: string, the function from `sklearn.metrics` to use.
      metric_dict_str: optional string, if not specified `metric_str` is used as
        the key in the returned dictionary.
      metric_post_process_fn: callable, if specified the final computed metric
        will be passed through this.
      **metric_fn_kwargs: kwargs, passed to the metric function we are calling.

    Returns:
      the function that calculates the metric in a dict.
    """
    if not hasattr(sklearn.metrics, metric_str):
        raise ValueError("sklearn.metrics does not have: %s" % metric_str)

    def fn(targets, predictions):
        metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return {metric_dict_str or metric_str: metric_val}
    return fn


def accuracy(targets, predictions):
    return {"accuracy": 100*sklearn.metrics.accuracy_score(targets, predictions)}


def f1_score_with_invalid(targets, predictions):
    """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.

    Args:
      targets: np.ndarray of targets, either 0 or 1
      predictions: np.ndarray of predictions, any integer value

    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    # For any prediction != 0 or 1, set it to the opposite of what the target is
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


def pearson_corrcoef(targets, predictions):
    """Pearson correlation coefficient."""
    return {"pearson_corrcoef":
            100 * scipy.stats.pearsonr(targets, predictions)[0]}


def spearman_corrcoef(targets, predictions):
    """Spearman correlation coefficient."""
    return {"spearman_corrcoef":
            100 * scipy.stats.spearmanr(targets, predictions)[0]}
