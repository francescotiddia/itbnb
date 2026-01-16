import numpy as np


def _safe_div(n, d):
    """Safe division: returns 0 if denominator is 0."""
    return np.divide(
        n, d, out=np.zeros_like(np.asarray(n, dtype=float)), where=(d != 0)
    )


def fpr(fp, tn):
    """False Positive Rate = FP / (FP + TN)"""
    return _safe_div(fp, fp + tn)


def tpr(tp, fn):
    """True Positive Rate (Recall) = TP / (TP + FN)"""
    return _safe_div(tp, tp + fn)


def fnr(tp, fn):
    """False Negative Rate (Type II error) = FN / (TP + FN)"""
    return _safe_div(fn, tp + fn)


def tnr(tn, fp):
    """True Negative Rate (Specificity) = TN / (TN + FP)"""
    return _safe_div(tn, tn + fp)


def precision(tp, fp):
    """Precision = TP / (TP + FP)"""
    return _safe_div(tp, tp + fp)


def mcc(tp, fp, tn, fn):
    """Matthews Correlation Coefficient"""
    tp = np.float64(tp)
    fp = np.float64(fp)
    tn = np.float64(tn)
    fn = np.float64(fn)

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if den <= 0:
        return 0.0

    return num / np.sqrt(den)


def f1_score(tp, fp, fn):
    """F1 = 2 * Precision * Recall / (Precision + Recall)"""
    prec = precision(tp, fp)
    rec = tpr(tp, fn)
    return _safe_div(2 * prec * rec, prec + rec)


def me(tp, fp, tn, fn):
    """Misclassification Error = (FP + FN) / total"""
    total = tp + fp + tn + fn
    return _safe_div(fp + fn, total)


def acc(tp, fp, tn, fn):
    """Accuracy = (TP+TN)/(TP+TN+FP+FN)"""
    return _safe_div(tp + tn, tp + tn + fp + fn)


def all_metrics_single(tp, fp, tn, fn):
    """Compute common binary classification metrics"""
    return {
        "precision": precision(tp, fp),
        "recall": tpr(tp, fn),
        "specificity": tnr(tn, fp),
        "fpr": fpr(fp, tn),
        "f1": f1_score(tp, fp, fn),
        "mcc": mcc(tp, fp, tn, fn),
        "misclassification_error": me(tp, fp, tn, fn),
        "fnr": fnr(tp, fn),
        "accuracy": acc(tp, fp, tn, fn),
    }


def all_metrics(tp, fp, tn, fn):
    """Compute common binary classification metrics from scalar confusion counts."""
    tp = tp.astype(np.float64, copy=False)
    fp = fp.astype(np.float64, copy=False)
    tn = tn.astype(np.float64, copy=False)
    fn = fn.astype(np.float64, copy=False)

    tp_fp = tp + fp
    tp_fn = tp + fn
    tn_fp = tn + fp
    total = tp + fp + tn + fn

    precision = np.divide(tp, tp_fp, out=np.zeros_like(tp), where=tp_fp != 0)
    recall = np.divide(tp, tp_fn, out=np.zeros_like(tp), where=tp_fn != 0)
    specificity = np.divide(tn, tn_fp, out=np.zeros_like(tp), where=tn_fp != 0)
    fpr = np.divide(fp, tn_fp, out=np.zeros_like(tp), where=tn_fp != 0)
    fnr = np.divide(fn, tp_fn, out=np.zeros_like(tp), where=tp_fn != 0)
    accuracy = np.divide(tp + tn, total, out=np.zeros_like(tp), where=total != 0)
    misclassification_error = np.divide(
        fp + fn, total, out=np.zeros_like(tp), where=total != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    num = tp * tn - fp * fn
    den = tp_fp * tp_fn * tn_fp * (tn + fn)
    mcc = np.divide(num, np.sqrt(den), out=np.zeros_like(tp), where=den > 0)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "accuracy": accuracy,
        "misclassification_error": misclassification_error,
        "f1": f1,
        "mcc": mcc,
    }


def metric_from_confusion(tp, fp, tn, fn, metric):
    return all_metrics(tp, fp, tn, fn)[metric]
