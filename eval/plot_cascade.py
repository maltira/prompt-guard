"""график компромисса каскадов: качество против доли запросов к судье

использование: uv run python -m eval.plot_cascade
слева val: для каждого каскада лучшая достижимая F1 по всем перебранным парам порогов и пара, выбранная по правилу
справа golden: каждый каскад прогнан один раз, рядом первая ступень отдельно (доля 0%) и один судья
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "figures"

# порядок и цвета категорий фиксированы; форма маркера дублирует цвет, у зелёного контраст ниже 3:1
SERIES = [
    ("embeddings-e5-base-mlp", "e5-base + MLP", "#2a78d6", "o"),
    ("embeddings-e5-large", "e5-large + логрегрессия", "#eb6834", "s"),
    ("lora-xlmr", "LoRA", "#1baf7a", "D"),
]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# смещение подписи точки на golden (пункты) и выравнивание: две точки эмбеддингов лежат рядом
GOLD_LABEL = {"embeddings-e5-base-mlp": (8, -16, "left"), "embeddings-e5-large": (-6, 12, "right"), "lora-xlmr": (8, 0, "left")}


def ru(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def overall_f1(path: Path) -> float:
    return next(r for r in json.loads(path.read_text(encoding="utf-8"))["slices"] if r["slice"] == "все")["f1"]


def frontier(grid: list[dict]) -> tuple[list[float], list[float]]:
    """верхняя огибающая: лучшая F1 среди пар порогов с долей запросов к судье не больше данной"""
    xs, ys, best = [], [], 0.0
    for row in sorted(grid, key=lambda r: (r["judge_share"], -(r["f1"] or 0))):
        if (row["f1"] or 0) > best:
            best = row["f1"]
            xs.append(row["judge_share"] * 100)
            ys.append(best)
    return xs, ys


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.set_xlabel("доля запросов, которые решает судья, %", color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ru(v)))


def main() -> None:
    fig, (ax_val, ax_gold) = plt.subplots(1, 2, figsize=(11.5, 4.7), facecolor=SURFACE)

    style(ax_val, "val: лучшая F1 при заданной доле запросов судье")
    ax_val.set_ylabel("F1", color=INK_2, fontsize=9)
    for name, label, color, marker in SERIES:
        sweep = json.loads((RESULTS_DIR / "cascade" / name / "val_sweep.json").read_text(encoding="utf-8"))
        xs, ys = frontier(sweep["grid"])
        ax_val.step(xs, ys, where="post", color=color, linewidth=2, label=label)
        ch = sweep["chosen"]
        ax_val.plot(ch["judge_share"] * 100, ch["f1"], marker, color=color, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        if name == "lora-xlmr":
            # вероятности LoRA почти двоичные: серая зона на val пуста, судье достаётся не больше 0,2% текстов
            ax_val.annotate(f"{label}: F1 {ru(ch['f1'], 3)}, серой зоны на val почти нет", (ch["judge_share"] * 100, ch["f1"]),
                            xytext=(12, -2), textcoords="offset points", color=INK, fontsize=9, va="center")
        else:
            ax_val.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            color=INK, fontsize=9, va="center")
    ax_val.set_xlim(-1.5, 70)
    ax_val.set_ylim(0.85, 1.005)
    ax_val.text(0.99, 0.04, "крупный маркер: пороги, выбранные по правилу", transform=ax_val.transAxes,
                ha="right", color=INK_2, fontsize=8)

    style(ax_gold, "golden: один прогон каждого каскада")
    for name, label, color, marker in SERIES:
        single = overall_f1(RESULTS_DIR / name / "golden.json")
        golden = json.loads((RESULTS_DIR / f"cascade-{name}" / "golden.json").read_text(encoding="utf-8"))
        share, f1 = golden["judged_share"] * 100, overall_f1(RESULTS_DIR / f"cascade-{name}" / "golden.json")
        ax_gold.plot([0, share], [single, f1], color=color, linewidth=2, label=label)
        ax_gold.plot(0, single, marker, color=SURFACE, markersize=9, markeredgecolor=color, markeredgewidth=2, zorder=5)
        ax_gold.plot(share, f1, marker, color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        dx, dy, ha = GOLD_LABEL[name]
        ax_gold.annotate(f"{label}\n{ru(f1)} при {ru(share, 1)}%", (share, f1), xytext=(dx, dy), textcoords="offset points",
                         color=INK, fontsize=8.5, va="center", ha=ha)
    judge = overall_f1(RESULTS_DIR / "judge-haiku" / "golden.json")
    ax_gold.axhline(judge, color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)))
    ax_gold.text(0.99, judge - 0.012, f"один судья: F1 {ru(judge)} при 100% запросов", transform=ax_gold.get_yaxis_transform(),
                 ha="right", va="top", color=INK_2, fontsize=8.5)
    ax_gold.set_xlim(-1, 30)
    ax_gold.set_ylim(0.7, 1.02)
    ax_gold.text(0.99, 0.04, "пустой маркер: первая ступень отдельно", transform=ax_gold.transAxes,
                 ha="right", color=INK_2, fontsize=8)

    fig.legend(*ax_val.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"cascade_tradeoff.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("сохранено:", OUT_DIR / "cascade_tradeoff.png")


if __name__ == "__main__":
    main()
