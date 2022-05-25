import numpy as np

def check_lengths(*arrays):
    """
    Check that all arrays have the same lenght.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked.
    """
    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        message = "Input arrays should all be the same length."
        raise ValueError(message)


def check_binaries(*arrays):
    """
    Check that all values in the arrays are 0s or 1s.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked.
    """
    values = [set(X) for X in arrays if X is not None]
    all_valid = all(v.issubset({0, 1}) for v in values)
    if not all_valid:
        message = "Input arrays should only contain 0s and/or 1s."
        raise ValueError(message)


def tp_rate(y_true, y_pred) -> float:
    """
    True positive rate.
    Parameters
    ----------
    y_true : 1d array-like of binaries
        Ground truth (correct) target values.
    y_pred : 1d array-like of binaries
        Estimated targets as returned by a classifier.
    """
    check_lengths(y_true, y_pred)
    check_binaries(y_true, y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if all(y_true == 0):
        return np.nan

    rate = (y_true @ y_pred) / (y_true @ y_true)
    return rate


def fp_rate(y_true, y_pred) -> float:
    """
    False positive rate.
    Parameters
    ----------
    y_true : 1d array-like of binaries
        Ground truth (correct) target values.
    y_pred : 1d array-like of binaries
        Estimated targets as returned by a classifier.
    """
    check_lengths(y_true, y_pred)
    check_binaries(y_true, y_pred)
    y_false = 1 - np.array(y_true)
    y_pred = np.array(y_pred)
    if all(y_false == 0):
        return np.nan

    rate = (y_false @ (y_pred)) / (y_false @ y_false)
    return rate


def classification_report(y_true, y_pred, A) -> str:
    """
    String showing the true positive rate and false
    positive rate for each group.
    Parameters
    ----------
    y_true : 1d array-like of binaries
        Ground truth (correct) target values.
    y_pred : 1d array-like of binaries
        Estimated targets as returned by a classifier.
    A: 1d array like
        Labels for the different groups.
    """
    check_lengths(y_true, y_pred, A)
    check_binaries(y_true, y_pred)
    groups = np.unique(A)
    header = "{:<30}{:^6}{:^6}".format("A", "TPR", "FPR")
    row_fmt = "{:<30}{:^6.2f}{:^6.2f}"
    lines = [header, "-" * len(header)]
    for g in groups:
        y_true_g = y_true[A == g]
        y_pred_g = y_pred[A == g]
        tpr_g = tp_rate(y_true_g, y_pred_g)
        fpr_g = fp_rate(y_true_g, y_pred_g)
        lines.append(row_fmt.format(g, tpr_g, fpr_g))

    tpr = tp_rate(y_true, y_pred)
    fpr = fp_rate(y_true, y_pred)
    lines.append(row_fmt.format("All", tpr, fpr))
    report = "\n".join(lines)
    return report