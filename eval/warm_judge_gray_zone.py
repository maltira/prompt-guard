"""добирает ответы судьи на тексты, которые первая ступень каскада считает неуверенными

судья вызывается только для текстов выборки, у которых вероятность атаки первой ступени
лежит в диапазоне [lo, hi]: вне него текст ни при каких порогах из этого диапазона судье не достанется
использование:
    uv run python -m eval.warm_judge_gray_zone --first-stage embeddings-e5-base-mlp --split val --lo 0.05 --hi 0.95
"""
import argparse
import json
from pathlib import Path

from data.schema import read_jsonl
from detectors.implementations.judge import JudgeDetector, MAX_TOKENS, wrap

RESULTS_DIR = Path("results")
SPLITS_DIR = Path("data/splits")


def cached(judge: JudgeDetector, text: str) -> bool:
    key = judge.llm._key(judge.system, wrap(text), None, MAX_TOKENS)
    return (judge.llm.cache_dir / f"{key}.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-stage", required=True, help="детектор первой ступени, его предсказания должны быть в results/")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--lo", type=float, required=True)
    parser.add_argument("--hi", type=float, required=True)
    args = parser.parse_args()

    scores = {}
    with open(RESULTS_DIR / args.first_stage / f"{args.split}_predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scores[row["id"]] = row["score"]
    recs = [r for r in read_jsonl(SPLITS_DIR / f"{args.split}.jsonl") if args.lo <= scores[r.id] <= args.hi]
    judge = JudgeDetector()
    todo = [r.text for r in recs if not cached(judge, r.text)]
    print(f"в диапазоне [{args.lo}, {args.hi}]: {len(recs)} текстов, уже есть ответ судьи: {len(recs) - len(todo)}, добираем: {len(todo)}", flush=True)
    judge.warm(todo)
    left = sum(not cached(judge, r.text) for r in recs)
    print(f"готово, без ответа осталось: {left}", flush=True)


if __name__ == "__main__":
    main()
