"""общий интерфейс детекторов: оценка, каскад и шлюз работают с любым из них"""
from abc import ABC, abstractmethod


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def predict(self, text: str) -> tuple[str, float]:
        """Возвращает (label, score): label - "attack" или "benign",
        score - вероятность атаки в [0, 1]."""

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]
