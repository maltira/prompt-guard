"""прогон детектора по выборке: метрики, срезы, задержка, сохранение результатов

использование:
    uv run python -m eval.run_eval --detector keyword --split val
    uv run python -m eval.run_eval --detector keyword --split golden

для разработки и подбора настроек используйте val. golden - только для итоговых цифр,
иначе он перестаёт быть честным экзаменом (см. data/README.md)
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from data.golden_hash import GoldenChangedError, verify_file
from data.schema import read_jsonl
from detectors.implementations.keyword import KeywordDetector
from eval.metrics import bootstrap_ci, scores

DETECTORS = {"keyword": KeywordDetector}
MIN_SLICE = 10  # категории меньше этого размера объединяются в одну строку
SPLITS_DIR = Path("data/splits")
RESULTS_DIR = Path("results")


def check_golden(path: Path) -> None:
    try:
        verify_file(path)
    except GoldenChangedError as e:
        raise SystemExit(f"остановлено: {e}")


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def report_row(name: str, y_true: list[bool], y_pred: list[bool]) -> dict:
    s = scores(y_true, y_pred)
    if not (s["attacks"] and s["benign"]):
        # precision и F1 имеют смысл только когда в срезе есть оба класса
        s["precision"] = s["f1"] = None
    row = {"slice": name, **s}
    if s["attacks"]:
        row["recall_ci"] = bootstrap_ci(y_true, y_pred, "recall")
    if s["benign"]:
        row["fpr_ci"] = bootstrap_ci(y_true, y_pred, "fpr")
    return row


def build_slices(recs, y_true, y_pred) -> list[dict]:
    groups = defaultdict(lambda: ([], []))
    keys = {
        "все": lambda r: "all",
        "язык": lambda r: f"lang={r.lang}",
        "категория": lambda r: f"category={r.category}",
        "происхождение": lambda r: f"source={r.source}",
    }
    rows = [report_row("все", y_true, y_pred)]
    for kind, key in keys.items():
        if kind == "все":
            continue
        groups = defaultdict(lambda: ([], []))
        for r, t, p in zip(recs, y_true, y_pred):
            g = groups[key(r)]
            g[0].append(t)
            g[1].append(p)
        small_t, small_p = [], []
        for name in sorted(groups):
            if kind == "категория" and len(groups[name][0]) < MIN_SLICE:
                small_t += groups[name][0]
                small_p += groups[name][1]
                continue
            rows.append(report_row(name, *groups[name]))
        if small_t:
            rows.append(report_row(f"category=прочие (меньше {MIN_SLICE} примеров)", small_t, small_p))
    return rows


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_ci(ci):
    return "-" if ci is None else f"[{ci[0]:.2f}; {ci[1]:.2f}]"


def print_table(rows: list[dict]) -> None:
    print("| срез | n | атак | безоп. | recall | 95% CI | precision | F1 | FPR | 95% CI |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['slice']} | {r['n']} | {r['attacks']} | {r['benign']} | {fmt(r['recall'])} | "
              f"{fmt_ci(r.get('recall_ci'))} | {fmt(r['precision'])} | {fmt(r['f1'])} | "
              f"{fmt(r['fpr'])} | {fmt_ci(r.get('fpr_ci'))} |")


def run(detector_name: str, split: str) -> dict:
    path = SPLITS_DIR / f"{split}.jsonl"
    if split == "golden":
        check_golden(path)
    recs = read_jsonl(path)
    detector = DETECTORS[detector_name]()

    latencies, preds = [], []
    for r in recs:
        t0 = time.perf_counter()
        label, score = detector.predict(r.text)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append((label, score))

    y_true = [r.label == "attack" for r in recs]
    y_pred = [label == "attack" for label, _ in preds]
    rows = build_slices(recs, y_true, y_pred)

    summary = {
        "detector": detector_name,
        "split": split,
        "n": len(recs),
        "latency_ms": {"p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
        "cost_per_1000_usd": getattr(detector, "cost_per_1000", None),
        "slices": rows,
    }

    out_dir = RESULTS_DIR / detector_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / f"{split}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, (label, score) in zip(recs, preds):
            f.write(json.dumps({"id": r.id, "label": r.label, "pred": label, "score": score,
                                "lang": r.lang, "category": r.category, "source": r.source},
                               ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    parser.add_argument("--split", default="golden", choices=["train", "val", "golden"])
    args = parser.parse_args()
    summary = run(args.detector, args.split)
    print(f"детектор: {summary['detector']}, выборка: {summary['split']}, n={summary['n']}")
    print(f"задержка на запрос, мс: p50={summary['latency_ms']['p50']:.4f}, "
          f"p95={summary['latency_ms']['p95']:.4f}")
    print_table(summary["slices"])


if __name__ == "__main__":
    main()
