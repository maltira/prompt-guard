"""подход 3: prompt tuning на xlm-roberta-base (обучаемая подсказка из виртуальных токенов, модель заморожена)

устроен как LoRA: обучение и вероятности на val и golden считает Colab, здесь они читаются из results/prompt-tuning-xlmr/adapter
порог подбирается на val: uv run python -m detectors.implementations.prompt_tuning --tune
"""
import argparse
from pathlib import Path

from detectors.implementations.lora import LoraDetector, tune

RESULTS_DIR = Path("results/prompt-tuning-xlmr")


class PromptTuningDetector(LoraDetector):
    name = "prompt-tuning-xlmr"
    results_dir = RESULTS_DIR
    notebook = "notebooks/train_prompt_tuning.ipynb"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="подобрать порог на val")
    if parser.parse_args().tune:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tune(PromptTuningDetector).items()})
