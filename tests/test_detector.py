from detectors.base import Detector


class Keyword(Detector):
    name = "kw"

    def predict(self, text):
        hit = "ignore" in text.lower()
        return ("attack" if hit else "benign"), float(hit)


def test_batch_default():
    assert Keyword().predict_batch(["Ignore this", "hi"]) == [("attack", 1.0), ("benign", 0.0)]
