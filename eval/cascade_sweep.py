"""подбор двух порогов каскада на val

использование: uv run python -m eval.cascade_sweep --first-stage embeddings-e5-base-mlp
первая ступень задаётся параметром, судья Haiku 5.5. Ответы судьи берутся только из кэша (запросов к API нет),
поэтому для всех текстов val с вероятностью первой ступени в [GRID_LO, GRID_HI] ответ судьи должен быть в кэше

правило выбора (задано до просмотра результатов): минимальная доля запросов к судье
при recall не ниже MIN_RECALL и доле ложных срабатываний не выше MAX_FPR; при равенстве лучше F1

запасное правило (задано до просмотра результатов на новом val): если ни одна точка не проходит основное,
берётся минимальная доля запросов к судье при recall не ниже recall самой широкой зоны минус FALLBACK_MARGIN
и FPR не выше FPR самой широкой зоны плюс FALLBACK_MARGIN
"""
import json
from pathlib import Path

from data.schema import read_jsonl
from detectors.implementations.judge import JudgeDetector
from eval.metrics import scores
from eval.warm_judge_gray_zone import cached

RESULTS_DIR = Path("results")
GRID_LO, GRID_HI = 0.05, 0.95
STEP = 0.05
MIN_RECALL = 0.97   # качество судьи на golden (0,996) минус около 3 п.п.
MAX_FPR = 0.03      # доля ложных срабатываний судьи (0,00) плюс около 3 п.п.
FALLBACK_MARGIN = 0.01  # допуск запасного правила относительно самой широкой серой зоны


def frange(a: float, b: float, step: float):
    n = round((b - a) / step)
    return [round(a + i * step, 2) for i in range(n + 1)]


def main(first_stage: str) -> None:
    val = read_jsonl("data/splits/val.jsonl")
    p = {}
    with open(RESULTS_DIR / first_stage / "val_predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            p[row["id"]] = row["score"]

    judge = JudgeDetector()
    judged = {}
    for r in val:
        if GRID_LO <= p[r.id] <= GRID_HI:
            if not cached(judge, r.text):
                raise SystemExit(f"нет ответа судьи для {r.id}, сначала python -m eval.warm_judge_gray_zone")
            judged[r.id] = judge.predict(r.text)[0] == "attack"
    y = [r.label == "attack" for r in val]
    n = len(val)

    rows = []
    for lo in frange(GRID_LO, 0.50, STEP):
        for hi in frange(0.50, GRID_HI, STEP):
            pred, n_judge = [], 0
            for r in val:
                s = p[r.id]
                if s < lo:
                    pred.append(False)
                elif s > hi:
                    pred.append(True)
                else:
                    pred.append(judged[r.id])
                    n_judge += 1
            m = scores(y, pred)
            rows.append({"lo": lo, "hi": hi, "judge_share": n_judge / n, "n_judged": n_judge,
                         "recall": m["recall"], "precision": m["precision"], "f1": m["f1"], "fpr": m["fpr"]})

    widest = next(r for r in rows if r["lo"] == GRID_LO and r["hi"] == GRID_HI)
    ok = [r for r in rows if r["recall"] >= MIN_RECALL and r["fpr"] <= MAX_FPR]
    rule_used = "основное"
    if not ok:
        rule_used = "запасное"
        ok = [r for r in rows if r["recall"] >= widest["recall"] - FALLBACK_MARGIN and r["fpr"] <= widest["fpr"] + FALLBACK_MARGIN]
    best = min(ok, key=lambda r: (r["judge_share"], -(r["f1"] or 0))) if ok else None
    first_only = next(r for r in rows if r["lo"] == 0.5 and r["hi"] == 0.5)

    out = RESULTS_DIR / "cascade" / first_stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_sweep.json").write_text(json.dumps(
        {"first_stage": first_stage, "rule": {"min_recall": MIN_RECALL, "max_fpr": MAX_FPR, "fallback_margin": FALLBACK_MARGIN, "rule_used": rule_used},
         "chosen": best, "widest_zone": widest, "first_stage_only": first_only, "grid": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if best:
        (out / "thresholds.json").write_text(json.dumps(
            {"first_stage": first_stage, "lo": best["lo"], "hi": best["hi"], "chosen_on": "val"},
            ensure_ascii=False, indent=2), encoding="utf-8")

    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"val: {n} текстов, ответы судьи на {len(judged)} из них (p в [{GRID_LO}, {GRID_HI}])")
    print(f"только первая ступень (порог 0,5): recall {f(first_only['recall'])}, FPR {f(first_only['fpr'])}, F1 {f(first_only['f1'])}")
    print(f"самая широкая серая зона [{GRID_LO}, {GRID_HI}]: recall {f(widest['recall'])}, FPR {f(widest['fpr'])}, F1 {f(widest['f1'])}, судье {widest['judge_share']:.1%}")
    print(f"\nправило: recall >= {MIN_RECALL}, FPR <= {MAX_FPR}, минимум доли запросов судье (применено: {rule_used})")
    print("лучшие точки по F1 для разных долей запросов судье:")
    print("| lo | hi | судье | recall | FPR | F1 |")
    print("|---|---|---|---|---|---|")
    seen = set()
    for r in sorted(rows, key=lambda r: (r["judge_share"], -(r["f1"] or 0))):
        key = round(r["judge_share"], 3)
        if key in seen:
            continue
        seen.add(key)
        mark = " **<-- выбрано**" if best and (r["lo"], r["hi"]) == (best["lo"], best["hi"]) else ""
        print(f"| {r['lo']:.2f} | {r['hi']:.2f} | {r['judge_share']:.1%} | {f(r['recall'])} | {f(r['fpr'])} | {f(r['f1'])} |{mark}")
    print("\nвыбрано:", best if best else "нет точки, удовлетворяющей правилу")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-stage", required=True, help="детектор первой ступени, его предсказания на val должны быть в results/")
    main(parser.parse_args().first_stage)
