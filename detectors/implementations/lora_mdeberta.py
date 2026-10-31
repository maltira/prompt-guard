"""LoRA на mdeberta-v3-base: те же настройки, что у xlm-roberta-base, меняется только базовая модель

устроен как LoRA: обучение и вероятности на val и golden считает Colab (notebooks/train_lora_mdeberta.ipynb),
здесь они читаются из results/lora-mdeberta/adapter
порог подбирается на val: uv run python -m detectors.implementations.lora_mdeberta --tune
"""
import argparse
from pathlib import Path

from detectors.implementations.lora import LoraDetector, tune

RESULTS_DIR = Path("results/lora-mdeberta")


class LoraMdebertaDetector(LoraDetector):
    name = "lora-mdeberta"
    results_dir = RESULTS_DIR
    notebook = "notebooks/train_lora_mdeberta.ipynb"
    base_model = "microsoft/mdeberta-v3-base"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="подобрать порог на val")
    if parser.parse_args().tune:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tune(LoraMdebertaDetector).items()})
