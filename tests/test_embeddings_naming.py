from detectors.implementations.embeddings import KINDS, detector_name


def test_detector_names():
    m = "intfloat/multilingual-e5-small"
    assert detector_name(m) == "embeddings-e5-small"
    assert detector_name(m, "logreg") == "embeddings-e5-small"
    assert detector_name(m, "knn") == "embeddings-e5-small-knn"
    assert detector_name("intfloat/multilingual-e5-large", "mlp") == "embeddings-e5-large-mlp"


def test_all_kinds_train_on_toy_data():
    import numpy as np
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(1, 0.3, (40, 8)), rng.normal(-1, 0.3, (40, 8))])
    y = np.array([True] * 40 + [False] * 40)
    for kind, (make, grid) in KINDS.items():
        params = {k: v[0] for k, v in grid.items()}
        clf = make(**params).fit(x, y)
        assert clf.predict_proba(x)[:, 1][y].mean() > 0.5, kind
