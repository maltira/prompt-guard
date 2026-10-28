"""бейзлайн: атака, если в тексте найдена типичная фраза инъекции"""
import re

from detectors.base import Detector

# фразы составлены по распространённым приёмам
PHRASES = [
    # английский
    "ignore previous instructions", "ignore all previous instructions", "ignore the above",
    "ignore your instructions", "disregard previous instructions", "disregard the above",
    "forget your instructions", "forget all previous instructions", "reveal your system prompt",
    "print your system prompt", "show your system prompt", "system override", "developer mode",
    "do anything now", "new instructions", "you are now", "pretend to be", "act as an unrestricted",
    # русский
    "игнорируй предыдущие инструкции", "игнорируй все предыдущие инструкции", "игнорируй указания выше",
    "забудь предыдущие инструкции", "забудь все инструкции", "забудь свои инструкции",
    "покажи системный промпт", "выведи системный промпт", "раскрой системный промпт",
    "системное переопределение", "режим разработчика", "новые инструкции", "ты теперь",
    "притворись", "действуй как",
]

_INVISIBLE = re.compile("[​‌‍﻿]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = _INVISIBLE.sub("", text).lower().replace("ё", "е")
    return _SPACES.sub(" ", text)


class KeywordDetector(Detector):
    name = "keyword"
    cost_per_1000 = 0.0

    def __init__(self, phrases: list[str] | None = None):
        self.phrases = [normalize(p) for p in (phrases or PHRASES)]

    def predict(self, text: str) -> tuple[str, float]:
        norm = normalize(text)
        hit = any(p in norm for p in self.phrases)
        return ("attack", 1.0) if hit else ("benign", 0.0)
