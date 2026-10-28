"""метрики бинарной детекции: recall, precision, F1, доля ложных срабатываний и доверительные интервалы"""
import random


def counts(y_true: list[bool], y_pred: list[bool]) -> tuple[int, int, int, int]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    return tp, fp, fn, tn


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


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
        "recall": recall, "precision": precision, "f1": f1, "fpr": _ratio(fp, fp + tn),
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
