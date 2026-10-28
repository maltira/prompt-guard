from eval.metrics import bootstrap_ci, scores


def test_scores_basic():
    y_true = [True, True, True, False, False]
    y_pred = [True, True, False, True, False]
    s = scores(y_true, y_pred)
    assert s["recall"] == 2 / 3
    assert s["precision"] == 2 / 3
    assert s["fpr"] == 1 / 2
    assert abs(s["f1"] - 2 / 3) < 1e-9


def test_scores_empty_class_is_none():
    s = scores([True, True], [True, False])
    assert s["fpr"] is None
    assert s["recall"] == 0.5


def test_no_positive_predictions():
    s = scores([True, False], [False, False])
    assert s["precision"] is None and s["f1"] is None and s["recall"] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    y_true = [True] * 50 + [False] * 50
    y_pred = [True] * 45 + [False] * 5 + [True] * 10 + [False] * 40
    lo, hi = bootstrap_ci(y_true, y_pred, "recall")
    assert lo <= 0.9 <= hi
    assert hi - lo < 0.25
