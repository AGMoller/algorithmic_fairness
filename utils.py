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

    
def perf_measure(y_actual:np.ndarray, y_hat:np.ndarray):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            tp += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            fp += 1
        if y_actual[i]==y_hat[i]==0:
            tn += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            fn += 1
            
    tp = tp / (tp + fn)
    fp = fp / (tn + fp)
    tn = tn / (tn + fp)
    fn = fn / (tp + fn)

    return tp, fp, tn, fn


def perf_measure(y_actual:np.ndarray, y_hat:np.ndarray):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            tp += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            fp += 1
        if y_actual[i]==y_hat[i]==0:
            tn += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            fn += 1
            
    tp = tp / (tp + fn)
    fp = fp / (tn + fp)
    tn = tn / (tn + fp)
    fn = fn / (tp + fn)

    return tp, fp, tn, fn

def classification_report(y_true, y_pred, A) -> str:
    """
    String showing the true positive, false
    positive, true negatve and false negative rate 
    for each group.
    Parameters
    ----------
    y_true : 1d array of binaries
        Ground truth (correct) target values.
    y_pred : 1d array of binaries
        Estimated targets as returned by a classifier.
    groups: 1d array
        Labels for the different groups.
    """
    check_lengths(y_true, y_pred, A)
    check_binaries(y_true, y_pred)
    groups = np.unique(A)
    header = "{:<30}{:^6}{:^6}{:^6}{:^6}".format("A", "TPR", "FPR", "TNR", "FNR")
    row_fmt = "{:<30}{:^6.2f}{:^6.2f}{:^6.2f}{:^6.2f}"
    lines = [header, "-" * len(header)]
    for group in groups:
        y_true_g = y_true[A == group]
        y_pred_g = y_pred[A == group]
        tp, fp, tn, fn = perf_measure(y_true_g, y_pred_g)
        lines.append(row_fmt.format(group, tp, fp, tn, fn))

    tp, fp, tn, fn = perf_measure(y_true, y_pred)
    lines.append(row_fmt.format("All", tp, fp, tn, fn))
    report = "\n".join(lines)
    
    return report