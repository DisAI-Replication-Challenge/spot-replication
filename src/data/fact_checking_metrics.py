import math
import numpy as np
import random
from scipy.stats import norm
from sklearn import metrics
import statistics


def MAP(y_true, y_pred, top_k=None):
    """
    Computes the mean average precision -- MAP and MAP@K.
    This function computes the mean average precision at k between two lists of
    items.
    Parameters
    ----------
    y_true : list
             A list of elements that are to be predicted (order doesn't matter)
    y_pred : list
             A list of predicted elements (order does matter)
    top_k : int
            The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision over the input lists
    """
    if len(y_pred) != len(y_true):
        raise ValueError('The length of y_pred and y_true should be equal')

    if top_k is None or top_k > y_true.shape[1]:
        top_k = y_true.shape[1]

    average_precision_scores = []
    for i, (label, score) in enumerate(zip(y_true, y_pred)):
        sorted_indices = np.argsort(score)[::-1]
        score = score[sorted_indices]
        label = label[sorted_indices]
        score[top_k:] = 0
        if sum(label) == 0:
            print(
                f"Skipping {i}th sample. No relevant documents in top {top_k} results")
            average_precision_score = 0
        else:
            average_precision_score = metrics.average_precision_score(
                label, score)
        average_precision_scores.append(average_precision_score)

    return np.mean(average_precision_scores), average_precision_scores


def recall(y_true, y_pred, top_k=1):
    if len(y_pred) != len(y_true):
        raise ValueError('The length of y_pred and y_true should be equal')

    if top_k is None or top_k > y_true.shape[1]:
        top_k = y_true.shape[1]

    recall_scores = []
    for i, (label, score) in enumerate(zip(y_true, y_pred)):
        sorted_indices = np.argsort(score)[::-1][:top_k]

        if sum(label) == 0:
            print(
                f"Skipping {i}th sample. No relevant documents in top {top_k} results")
            recall_score = 0
        else:
            max_score = np.sum(label)
            recall_score = np.sum(label[sorted_indices]) / max_score
        recall_scores.append(recall_score)

    return np.mean(recall_scores), recall_scores


def precision(y_true, y_pred, top_k=1):
    if len(y_pred) != len(y_true):
        raise ValueError('The length of y_pred and y_true should be equal')

    if top_k is None or top_k > y_true.shape[1]:
        top_k = y_true.shape[1]

    precision_scores = []
    for i, (label, score) in enumerate(zip(y_true, y_pred)):
        sorted_indices = np.argsort(score)[::-1][:top_k]

        precision_score = np.sum(label[sorted_indices]) / top_k
        precision_scores.append(precision_score)

    return np.mean(precision_scores), precision_scores


def F1(y_true, y_pred, top_k=1):
    if len(y_pred) != len(y_true):
        raise ValueError('The length of y_pred and y_true should be equal')

    if top_k is None or top_k > y_true.shape[1]:
        top_k = y_true.shape[1]

    precision_scores = []
    recall_scores = []
    f1_scores = []
    for i, (label, score) in enumerate(zip(y_true, y_pred)):
        sorted_indices = np.argsort(score)[::-1][:top_k]

        precision_score = np.sum(label[sorted_indices]) / top_k
        recall_score = np.sum(label[sorted_indices]) / np.sum(label)
        f1_score = 2 * precision_score * recall_score / \
            (precision_score + recall_score)

        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)

    return np.mean(f1_scores), f1_scores


def MRR(y_true, y_pred, top_k=1):
    if len(y_pred) != len(y_true):
        raise ValueError('The length of y_pred and y_true should be equal')

    MRRs = []
    for i, (label, score) in enumerate(zip(y_true, y_pred)):
        sorted_indices = np.argsort(score)[::-1]
        sorted_label = label[sorted_indices]
        true_positions = np.where(sorted_label)[0]
        if len(true_positions) == 0:
            print(
                f"Skipping {i}th sample. No relevant documents in top {top_k} results")
            MRR = 0
        else:
            MRR = 1 / (true_positions[0] + 1)
        MRRs.append(MRR)

    return np.mean(MRRs), MRRs


def has_positives(y_true, y_pred, top_k=1):
    if len(y_pred) != len(y_true):
        raise ValueError('The length of y_pred and y_true should be equal')

    if top_k is None or top_k > y_true.shape[1]:
        top_k = y_true.shape[1]

    has_positives = []
    for i, (label, score) in enumerate(zip(y_true, y_pred)):
        sorted_indices = np.argsort(score)[::-1][:top_k]
        if sum(label) == 0:
            print(
                f"Skipping {i}th sample. No relevant documents in top {top_k} results")
            has_positive = 0
        else:
            has_positive = 1 if label[sorted_indices].sum() else 0
        has_positives.append(has_positive)

    return np.mean(has_positives), has_positives


def binary_ci(success, total, alpha=0.95):
    """
    Using Agresti-Coull interval

    Return mean and confidence interval (lower and upper bound)
    """
    z = statistics.NormalDist().inv_cdf((1 + alpha) / 2)
    total = total + z**2
    loc = (success + (z**2) / 2) / total
    diameter = z * math.sqrt(loc * (1 - loc) / total)
    return loc, loc - diameter, loc + diameter


def bootstrap_ci(scores, alpha=0.95):
    """
    Bootstrapping based estimate.

    Return mean and confidence interval (lower and upper bound)
    """
    loc, scale = norm.fit(scores)
    bootstrap = [sum(random.choices(scores, k=len(scores))) /
                 len(scores) for _ in range(1000)]
    lower, upper = norm.interval(alpha, *norm.fit(bootstrap))

    return loc, lower, upper


def pair_success_at_k(ranks, k=10):
    """
    Pair S@K - How many fact-check-post pairs from all the pairs ended up in the top K.
    """
    values = [rank <= k for query in ranks for rank in query]
    return binary_ci(sum(values), len(values))


def post_success_at_k(ranks, k=10):
    """
    Post S@K - For how many posts at least one pair ended up in the top K.
    """
    values = [any(rank <= k for rank in query) for query in ranks]
    return binary_ci(sum(values), len(values))


def precision_at_k(ranks, k=10):
    """
    P@K - How many positive hits in the top K
    """
    values = [sum(rank <= k for rank in query) for query in ranks]
    return binary_ci(sum(values), len(values) * k)


def mrr(ranks):
    """
    Mean Reciprocal Rank: 1/r for r in ranks
    """
    values = [1 / min(query) for query in ranks]
    return bootstrap_ci(values)


def map_(ranks):
    """
    Mean Average Precision: As defined here page 7: https://datascience-intro.github.io/1MS041-2022/Files/AveragePrecision.pdf
    """
    values = [
        np.mean([
            (i + 1) / rank
            for i, rank in enumerate(sorted(query))
        ])
        for query in ranks
    ]
    return bootstrap_ci(values)


def map_k(ranks, k=5):
    values = []
    for query in ranks:
        ap_at_k = 0
        num_correct = 0
        for i, rank in enumerate(query):
            if rank <= k:
                num_correct += 1
                ap_at_k += num_correct / (i+1)
        values.append(ap_at_k)
    return bootstrap_ci(values)


def standard_metrics(ranks):
    """
    Calculate several metrics and their CIs based on the ranks provided

    Attributes:
        ranks - Iterable of results for individual queries. For each query a list of ranks is expected.
    """

    return {
        'pair_success_at_10': pair_success_at_k(ranks),
        'post_success_at_10': post_success_at_k(ranks),
        'precision_at_10': precision_at_k(ranks),
        'mrr': mrr(ranks),
        'map': map_(ranks),
        'map_5': map_k(ranks),
    }
