"""метрики бинарной детекции: recall, precision, F1, MCC, доля ложных срабатываний, ROC AUC и доверительные интервалы"""
import math
import random

import numpy as np
from scipy.stats import rankdata


def counts(y_true: list[bool], y_pred: list[bool]) -> tuple[int, int, int, int]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    return tp, fp, fn, tn


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """коэффициент корреляции Мэтьюса: от -1 до 1, учитывает все четыре ячейки матрицы ошибок"""
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(d) if d else None


def roc_auc(y_true: list[bool], y_score: list[float]) -> float | None:
    """ROC AUC: вероятность, что случайная атака получит оценку выше случайного безопасного текста (при равных оценках засчитывается половина)"""
    y = np.asarray(y_true, dtype=bool)
    sc = np.asarray(y_score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(sc)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(y_true: list[bool], y_score: list[float], n_boot: int = 1000, seed: int = 13) -> tuple[float, float] | None:
    """95% доверительный интервал ROC AUC методом бутстрэпа по примерам"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=bool)
    sc = np.asarray(y_score, dtype=float)
    n = len(y)
    values = []
    for _ in range(n_boot):
        ids = rng.integers(0, n, n)
        v = roc_auc(y[ids].tolist(), sc[ids].tolist())
        if v is not None:
            values.append(v)
    if len(values) < n_boot // 2:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def scores(y_true: list[bool], y_pred: list[bool]) -> dict:
    """None означает, что знаменатель нулевой (например, нет безопасных примеров в срезе)"""
    tp, fp, fn, tn = counts(y_true, y_pred)
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    if recall is None or precision is None or recall + precision == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "n": len(y_true), "attacks": tp + fn, "benign": fp + tn,
        "recall": recall, "precision": precision, "f1": f1, "mcc": mcc(tp, fp, fn, tn), "fpr": _ratio(fp, fp + tn),
    }


def bootstrap_ci(y_true: list[bool], y_pred: list[bool], key: str,
                 n_boot: int = 1000, seed: int = 13) -> tuple[float, float] | None:
    """95% доверительный интервал метрики методом бутстрэпа по примерам"""
    rng = random.Random(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        ids = rng.choices(range(n), k=n)
        v = scores([y_true[i] for i in ids], [y_pred[i] for i in ids])[key]
        if v is not None:
            values.append(v)
    if len(values) < n_boot // 2:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]
