"""дописывает MCC и ROC AUC в готовые results/**/<выборка>.json по сохранённым предсказаниям

детекторы, модели и судья заново не запускаются: берутся `<выборка>_predictions.jsonl`
MCC считается для каждого среза, ROC AUC и его интервал только для строки «все» и только у детекторов, не являющихся каскадами
использование: uv run python -m eval.add_metrics
"""
import json
from pathlib import Path

from eval.metrics import bootstrap_auc_ci, roc_auc, scores

RESULTS_DIR = Path("results")


def slice_rows(preds: list[dict], name: str) -> list[dict] | None:
    if name == "все":
        return preds
    key, _, value = name.partition("=")
    if key in ("lang", "category", "source") and not value.startswith("прочие"):
        return [r for r in preds if r[key] == value]
    return None


def main() -> None:
    updated = 0
    for path in sorted(RESULTS_DIR.glob("**/*.json")):
        pred_path = path.with_name(path.stem + "_predictions.jsonl")
        if path.stem not in ("val", "golden") or not pred_path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "slices" not in summary:
            continue
        preds = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cascade = summary.get("cascade_thresholds") is not None
        for row in summary["slices"]:
            sub = slice_rows(preds, row["slice"])
            if sub is None:
                continue
            y_true = [r["label"] == "attack" for r in sub]
            y_pred = [r["pred"] == "attack" for r in sub]
            row["mcc"] = scores(y_true, y_pred)["mcc"]
            if row["slice"] == "все" and not cascade:
                y_score = [r["score"] for r in sub]
                row["roc_auc"] = roc_auc(y_true, y_score)
                row["roc_auc_ci"] = bootstrap_auc_ci(y_true, y_score)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1
    print(f"обновлено файлов: {updated}")


if __name__ == "__main__":
    main()
