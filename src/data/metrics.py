from rouge_score import rouge_scorer
from rouge_score import scoring
import sacrebleu
from sklearn.metrics import f1_score
import sklearn.metrics
import scipy.stats
import numpy as np
import collections
import re

import data.qa_utils as qa_utils


def bleu(targets, predictions, tokenizer="intl"):
    """Computes BLEU score.

    Args:
      targets: list of strings or list of list of strings if multiple references
        are present.
      predictions: list of strings
      tokenizer: tokenizer option for corpus_bleu

    Returns:
      bleu_score across all targets and predictions
    """
    if isinstance(targets[0], list):
        targets = [[x for x in target] for target in targets]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        targets = [targets]

    bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                       smooth_method="exp",
                                       smooth_value=0.0,
                                       force=False,
                                       lowercase=False,
                                       tokenize=tokenizer,
                                       use_effective_order=False)
    return {"bleu": bleu_score.score}


def _prepare_summary_rouge(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary


def rouge(
    targets,
    predictions,
    score_keys=("rouge1", "rouge2", "rougeLsum"),
    **kwargs,
):
    """Computes rouge score nondeterministically using the bootstrap.

    Args:
      targets: list of strings
      predictions: list of strings
      score_keys: list of strings with the keys to compute.
      **kwargs: additional keyword arguments for RougeScorer.

    Returns:
      dict with score_key: rouge score across all targets and predictions
    """

    scorer = rouge_scorer.RougeScorer(rouge_types=score_keys, **kwargs)
    aggregator = scoring.BootstrapAggregator()

    for prediction, target in zip(predictions, targets):
        target = _prepare_summary_rouge(target)
        prediction = _prepare_summary_rouge(prediction)
        aggregator.add_scores(scorer.score(
            target=target, prediction=prediction))
    result = aggregator.aggregate()
    # for key in score_keys:
    #     logging.info(
    #         "%s = %.2f, 95%% confidence [%.2f, %.2f]",
    #         key,
    #         result[key].mid.fmeasure*100,
    #         result[key].low.fmeasure*100,
    #         result[key].high.fmeasure*100,
    #     )
    return {key: result[key].mid.fmeasure*100 for key in score_keys}


def rouge_mean(
    targets,
    predictions,
    score_keys=("rouge1", "rouge2", "rougeLsum"),
    **kwargs,
):
    """Computes rouge score deterministically (no bootstrap).

    Args:
      targets: list of strings
      predictions: list of strings
      score_keys: list of strings with the keys to compute
      **kwargs: additional keyword arguments for RougeScorer.

    Returns:
      dict with score_key: rouge score across all targets and predictions
    """

    scorer = rouge_scorer.RougeScorer(rouge_types=score_keys, **kwargs)
    count = 0
    sum_scores = collections.defaultdict(float)
    for prediction, target in zip(predictions, targets):
        target = _prepare_summary_rouge(target)
        prediction = _prepare_summary_rouge(prediction)
        scores = scorer.score(target=target, prediction=prediction)
        count += 1
        for k, v in scores.items():
            sum_scores[k] += v.fmeasure
    if count == 0:
        raise ValueError(
            "Predictions and targets must both have nonzero length")
    result = {k: v / count for k, v in sum_scores.items()}
    return {key: result[key] * 100 for key in score_keys}


def span_squad(targets, predictions):
    """Computes SQuAD metrics for span prediction tasks.

    Uses qa metric function to compute EM and F1 score.

    Args:
      targets: list of dict of answers (list of strings) and context (string)
      predictions: list of strings, each string is contains the space tokenized
        ids in the format: "start: 3 end: 6"

    Returns:
      dict with score_key: squad score across all targets and predictions
    """
    assert len(targets) == len(predictions)

    def space_tok(s):
        return re.sub(r"\W", " ", s).split()

    def get_answer_text_from_context(context, answer_tokens):
        """Find the answer in the context given the answer tokens."""
        # In the initial training iterations, the model can output garbage.
        # Returning an empty string in such cases.
        if len(answer_tokens) < 4:
            return ""

        # Model sometimes predicts words instead of numbers in the answer. Return
        # an empty string in that case.
        try:
            start_index = int(answer_tokens[1])
            end_index = int(answer_tokens[3])
        except ValueError:
            return ""

        return " ".join(context[start_index:end_index+1])

    contexts = [space_tok(t["context"]) for t in targets]
    answers = [t["answers"] for t in targets]

    predictions = [space_tok(p) for p in predictions]
    final_predictions = [
        get_answer_text_from_context(c, p) for c, p in zip(contexts, predictions)
    ]

    return squad(answers, final_predictions)


def squad(targets, predictions):
    """Computes SQuAD metrics, maximizing over answers per question.

    Args:
      targets: list of lists of strings
      predictions: list of strings

    Returns:
      dict with score_key: squad score across all targets and predictions
    """
    if type(targets[0]) is list:
        targets = [[qa_utils.normalize_squad(t) for t in u] for u in targets]
    else:
        targets = [[qa_utils.normalize_squad(u)] for u in targets]

    predictions = [qa_utils.normalize_squad(p) for p in predictions]
    return qa_utils.qa_metrics(targets, predictions)


def trivia_qa(targets, predictions):
    """Computes TriviaQA metrics, maximizing over answers per question.

    Args:
      targets: list of lists of strings
      predictions: list of strings

    Returns:
      dict with score_key: squad score across all targets and predictions
    """
    targets = [[qa_utils.normalize_trivia_qa(t) for t in u] for u in targets]
    predictions = [qa_utils.normalize_trivia_qa(p) for p in predictions]
    return qa_utils.qa_metrics(targets, predictions)


def accuracy(targets, predictions):
    return {"accuracy": 100*sklearn.metrics.accuracy_score(targets, predictions)}


def sequence_accuracy(targets, predictions):
    """Computes per-sequence accuracy.

    For each example, returns 1.0 if the target sequence EXACTLY matches the
    predicted sequence. Else, 0.0.

    Args:
      targets: list of strings
      predictions: list of strings

    Returns:
      float. Average sequence-level accuracy.
    """
    assert len(targets) == len(predictions)
    seq_acc = 100 * np.mean([p == t for p, t in zip(predictions, targets)])
    return {"sequence_accuracy": seq_acc}


def pearson_corrcoef(targets, predictions):
    """Pearson correlation coefficient."""
    # convert targets and predictions arrays to float
    targets = np.asarray(targets, dtype=np.float16)
    predictions = np.asarray(predictions, dtype=np.float16)
    return {"pearson_corrcoef":
            100 * scipy.stats.pearsonr(targets, predictions)[0]}


def spearman_corrcoef(targets, predictions):
    """Spearman correlation coefficient."""
    # convert targets and predictions to float
    targets = np.asarray(targets, dtype=np.float16)
    predictions = np.asarray(predictions, dtype=np.float16)
    return {"spearman_corrcoef":
            100 * scipy.stats.spearmanr(targets, predictions)[0]}


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
    """Computes the unweighted average of the F1 per class."""
    return sklearn_metrics_wrapper(
        "fbeta_score",
        metric_dict_str="mean_%dclass_f1" % num_classes,
        metric_post_process_fn=lambda x: 100 * x,
        beta=1,
        labels=range(num_classes),
        average="macro",
        **metric_fn_kwargs)


def all_match(targets, predictions):
    """Computes whether all targets match all predictions exactly."""
    return {"exact_match": 100 * float(np.array_equal(targets, predictions))}


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
    try:
        predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
        return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}
    except Exception as e:
        print(e)
        print(targets, predictions),
        return {"f1": 0.0}


def multirc_f1_over_all_answers(targets, predictions):
    """Special metric for MultiRC which computes F1 score over all examples.

    This is necessary because the targets/predictions for MultiRC are dicts and
    the f1_score_with_invalid expects a list of True/False labels, not dicts. As
    a result we just need to key in the "value" for each of the example dicts
    before feeding into f1_score_with_invalid.

    Args:
      targets: list of dicts, where each dict has a "value" key.
      predictions: list of dicts, where each dict has a "value" key.

    Returns:
      F1 score over values, where any prediction != 0 or 1 is counted as wrong.
    """
    return f1_score_with_invalid(
        [t["value"] for t in targets], [p["value"] for p in predictions]
    )


def auc(targets, predictions, targets_threshold=None):
    """Compute Area Under the ROC and PR curves.

    ROC - Receiver Operating Characteristic
    PR  - Precision and Recall

    Args:
      targets: np.ndarray of targets, either 0 or 1, or continuous values.
      predictions: np.ndarray of predictions, any value.
      targets_threshold: float, if target values are continuous values, this
        threshold binarizes them.

    Returns:
      A dictionary with AUC-ROC and AUC-PR scores.
    """

    if targets_threshold is not None:
        targets = np.array(targets)
        targets = np.where(targets < targets_threshold,
                           np.zeros_like(targets, dtype=np.int32),
                           np.ones_like(targets, dtype=np.int32))

    return {
        "auc-roc": sklearn.metrics.roc_auc_score(targets, predictions),
        "auc-pr": sklearn.metrics.average_precision_score(targets, predictions),
    }


def score_auc(targets, scores, targets_threshold=None):
    """Compute Area Under the ROC and PR curves.

    ROC - Receiver Operating Characteristic
    PR  - Precision and Recall

    Args:
      targets: np.ndarray of targets, either 0 or 1, or continuous values.
      scores: np.ndarray of scores, any value.
      targets_threshold: float, if target values are continuous values, this
        threshold binarizes them.

    Returns:
      A dictionary with AUC-ROC and AUC-PR scores.
    """

    return auc(
        targets=targets, predictions=scores, targets_threshold=targets_threshold)


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
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


def mean_group_metric(metric_fn,
                      group_key="group",
                      value_key="value",
                      return_subgroup_scores=False):
    """Returns a metric that averages `metric_fn` on sub-groups of results.

    The sub-groups are defined by aggregating results (targets and predictions)
    by accessing the feature specified by `group_key` in the target dicts.

    **WARNING**: Using this function can produce unreliable results if you do not
    pass in full groups. For example, if you evaluate over a random subsample of a
    validation set and do not retain all of the examples in each group, you may
    get results which aren't directly comparable to using the full validation set.

    Args:
      metric_fn: function, the metric to compute on the subgroups.
      group_key: string, the key for the grouping value in the target dictionary.
      value_key: string, the key for the value in the dictionaries.
      return_subgroup_scores: If true, include the scores for each sub-group.
    """
    def my_metric(targets, predictions):
        """Computes mean of `metric_fn` over subgroups of results."""
        grouped_values = collections.defaultdict(lambda: ([], []))
        for targ, pred in zip(targets, predictions):
            g = targ[group_key]
            grouped_values[g][0].append(targ[value_key])
            grouped_values[g][1].append(pred[value_key])
        group_scores = collections.defaultdict(list)
        for group, (targets, predictions) in grouped_values.items():
            for metric, score in metric_fn(targets, predictions).items():
                group_scores[metric].append(score)
                if return_subgroup_scores:
                    group_scores["%s-%s" % (group, metric)].append(score)
        return {metric: np.mean(scores) for metric, scores in group_scores.items()}
    return my_metric


def deduplicate_metric(metric_fn,
                       group_key: str = "group",
                       value_key: str = "value"):
    """Returns a metric that only considers one example per group.

    Useful for things like ReCoRD where inputs may be replicated during training
    to handle multiple labels, but where at eval we only want a single copy of
    each example.

    Args:
      metric_fn: function, the metric to compute on the unique examples.
      group_key: the key for the grouping value in the target dictionary.
      value_key: the key for the value in the dictionaries.

    Returns:
      A metric function that deduplicated based on the grouping key before
      returning a metric.
    """
    def _deduplicated_metric(targets, predictions):
        """Deduplicate targets and predictions and pass that to the metric fn."""
        processed_groups = set()
        deduplicated_targets = []
        deduplicated_predictions = []
        for targ, pred in zip(targets, predictions):
            group = targ[group_key]
            if group in processed_groups:
                continue
            processed_groups.add(group)
            deduplicated_targets.append(targ[value_key])
            deduplicated_predictions.append(pred[value_key])
        return metric_fn(deduplicated_targets, deduplicated_predictions)
    return _deduplicated_metric


def macro_f1(targets, predictions):
    """Computes macro F1 score.

    Args:
      targets: list of lists of strings
      predictions: list of strings

    Returns:
      dict with score_key: macro f1 score across all targets and predictions
    """
    score = f1_score(targets, predictions, average='macro')
    return {"macro_f1": 100 * score}
