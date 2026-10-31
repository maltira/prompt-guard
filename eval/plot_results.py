"""графики результатов на golden: тепловая карта эмбеддингов, точечный график качества и столбцы F1

использование: uv run python -m eval.plot_results
все значения читаются из results/*/golden.json, те же файлы, что дают таблицы в RESULTS.md
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "figures"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
EMB, LORA, CASCADE, JUDGE = "#2a78d6", "#1baf7a", "#52514e", "#eb6834"
KEYWORD, PROMPT = "#9ca3af", "#b45309"

SIZES = [("small", "e5-small"), ("base", "e5-base"), ("large", "e5-large")]
CLASSIFIERS = [("", "логрегрессия"), ("-knn", "kNN"), ("-mlp", "MLP")]

# смещение подписи в пунктах (dx, dy) и выравнивание: точки эмбеддингов лежат близко друг к другу
OFFSET = {
    "e5-small + логрегрессия": (6, -10, "left"),
    "e5-small + MLP": (6, 6, "left"),
    "e5-small + kNN": (6, -4, "left"),
    "e5-base + логрегрессия": (-8, -4, "right"),
    "e5-base + MLP": (0, -12, "center"),
    "e5-base + kNN": (6, -6, "left"),
    "e5-large + логрегрессия": (-8, -2, "right"),
    "e5-large + MLP": (-8, -10, "right"),
    "e5-large + kNN": (-8, 8, "right"),
    "LoRA": (10, 4, "left"),
    "каскад LoRA → судья": (6, -14, "left"),
    "каскад e5-base → судья": (6, 6, "left"),
    "каскад e5-large → судья": (-8, -8, "right"),
    "судья Haiku 5.5": (0, 12, "center"),
    "ключевые слова": (8, 0, "left"),
    "prompt tuning, 1": (-8, -10, "right"),
    "prompt tuning, 2": (-8, 6, "right"),
}


def ru(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def golden(name: str) -> dict:
    data = json.loads((RESULTS_DIR / name / "golden.json").read_text(encoding="utf-8"))
    return next(r for r in data["slices"] if r["slice"] == "все")


def style(fig, ax) -> None:
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def heatmap(out: Path) -> None:
    f1 = np.array([[golden(f"embeddings-e5-{s}{c}")["f1"] for c, _ in CLASSIFIERS] for s, _ in SIZES])
    fpr = np.array([[golden(f"embeddings-e5-{s}{c}")["fpr"] for c, _ in CLASSIFIERS] for s, _ in SIZES])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    style(fig, ax)
    im = ax.imshow(f1, cmap="Blues", vmin=0.6, vmax=0.85)
    for i in range(len(SIZES)):
        for j in range(len(CLASSIFIERS)):
            ax.text(j, i, f"F1 {ru(f1[i, j])}\nFPR {ru(fpr[i, j])}", ha="center", va="center",
                    color="white" if f1[i, j] > 0.75 else INK, fontsize=10)
    ax.set_xticks(range(len(CLASSIFIERS)))
    ax.set_xticklabels([n for _, n in CLASSIFIERS])
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels([n for _, n in SIZES])
    ax.spines[:].set_visible(False)
    ax.set_title("Эмбеддинги на golden: модель × классификатор", loc="left", color=INK, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("F1", color=INK_2)
    cbar.outline.set_visible(False)
    save(fig, out)


def f1_bars(out: Path) -> None:
    rows = []
    for s, sn in SIZES:
        for c, cn in CLASSIFIERS:
            rows.append((f"{sn} + {cn}", golden(f"embeddings-e5-{s}{c}")["f1"], EMB))
    rows.append(("LoRA xlm-roberta-base", golden("lora-xlmr")["f1"], LORA))
    rows.append(("каскад LoRA → судья", golden("cascade-lora-xlmr")["f1"], CASCADE))
    rows.append(("каскад e5-large → судья", golden("cascade-embeddings-e5-large")["f1"], CASCADE))
    rows.append(("каскад e5-base → судья", golden("cascade-embeddings-e5-base-mlp")["f1"], CASCADE))
    rows.append(("судья Haiku 5.5", golden("judge-haiku")["f1"], JUDGE))
    rows.append(("ключевые слова", golden("keyword")["f1"], KEYWORD))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 6.2))
    style(fig, ax)
    names, vals, colors = zip(*rows)
    bars = ax.barh(names, vals, color=colors, height=0.66)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, ru(v), va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 на golden", color=INK_2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
    ax.set_title("F1 на golden по подходам", loc="left", color=INK, fontsize=12)
    save(fig, out)


def points() -> list[dict]:
    """все подходы на golden: FPR по оси X, recall по оси Y"""
    rows = []
    for s, sn in SIZES:
        for c, cn in CLASSIFIERS:
            r = golden(f"embeddings-e5-{s}{c}")
            rows.append({"label": f"{sn} + {cn}", "fpr": r["fpr"], "recall": r["recall"], "color": EMB, "marker": "o"})
    lora = golden("lora-xlmr")
    rows.append({"label": "LoRA", "fpr": lora["fpr"], "recall": lora["recall"], "color": LORA, "marker": "*", "size": 220})
    for name, label in [("cascade-lora-xlmr", "каскад LoRA → судья"),
                        ("cascade-embeddings-e5-base-mlp", "каскад e5-base → судья"),
                        ("cascade-embeddings-e5-large", "каскад e5-large → судья")]:
        r = golden(name)
        rows.append({"label": label, "fpr": r["fpr"], "recall": r["recall"], "color": CASCADE, "marker": "^"})
    judge = golden("judge-haiku")
    rows.append({"label": "судья Haiku 5.5", "fpr": judge["fpr"], "recall": judge["recall"], "color": JUDGE, "marker": "P"})
    kw = golden("keyword")
    rows.append({"label": "ключевые слова", "fpr": kw["fpr"], "recall": kw["recall"], "color": KEYWORD, "marker": "x", "filled": False})
    for run in ("run1", "run2"):
        pt = golden(f"prompt-tuning-xlmr/{run}")
        rows.append({"label": f"prompt tuning, {run[-1]}", "fpr": pt["fpr"], "recall": pt["recall"], "color": PROMPT, "marker": "X"})
    return rows


def quality_map(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    style(fig, ax)
    for p in points():
        size = p.get("size", 90)
        if p.get("filled", True):
            ax.scatter(p["fpr"], p["recall"], s=size, color=p["color"], marker=p["marker"],
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
        else:
            ax.scatter(p["fpr"], p["recall"], s=size, color=p["color"], marker=p["marker"], linewidth=1.5, zorder=3)
        dx, dy, ha = OFFSET[p["label"]]
        ax.annotate(p["label"], (p["fpr"], p["recall"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("доля ложных срабатываний (FPR), левее лучше", color=INK_2)
    ax.set_ylabel("полнота (recall), выше лучше", color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Качество на golden: какие атаки ловим и сколько безопасных задеваем", loc="left", color=INK, fontsize=12)
    handles = [
        Line2D([], [], marker="o", color=EMB, linestyle="", label="эмбеддинги + классификатор"),
        Line2D([], [], marker="*", color=LORA, linestyle="", markersize=12, label="LoRA"),
        Line2D([], [], marker="^", color=CASCADE, linestyle="", label="каскады"),
        Line2D([], [], marker="P", color=JUDGE, linestyle="", label="судья Haiku 5.5"),
        Line2D([], [], marker="X", color=PROMPT, linestyle="", label="prompt tuning (три запуска, не обучился)"),
        Line2D([], [], marker="x", color=KEYWORD, linestyle="", label="ключевые слова"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    save(fig, out)


def benign_fpr(out: Path) -> None:
    """доля ложных срабатываний на обычных и пограничных безопасных письмах golden"""
    detectors = [("embeddings-e5-base-mlp", "e5-base + MLP", EMB), ("lora-xlmr", "LoRA", LORA),
                 ("cascade-lora-xlmr", "каскад LoRA → судья", CASCADE)]
    groups = [("category=benign", "обычные безопасные"), ("category=benign_borderline", "пограничные безопасные")]
    fig, ax = plt.subplots(figsize=(8, 5))
    style(fig, ax)
    width, x = 0.26, np.arange(len(groups))
    for i, (name, label, color) in enumerate(detectors):
        slices = {r["slice"]: r for r in json.loads((RESULTS_DIR / name / "golden.json").read_text(encoding="utf-8"))["slices"]}
        vals = [slices[key]["fpr"] for key, _ in groups]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, ru(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([g for _, g in groups])
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("доля ложных срабатываний (FPR)", color=INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title("Безопасные письма: где детекторы ошибаются", loc="left", color=INK, fontsize=12)
    save(fig, out)


def cost_f1(out: Path) -> None:
    """качество против цены: локальные модели стоят 0, их ставим у левого края шкалы"""
    items = [("судья Haiku 5.5", "judge-haiku", JUDGE, "P"),
             ("каскад e5-base → судья", "cascade-embeddings-e5-base-mlp", CASCADE, "^"),
             ("каскад e5-large → судья", "cascade-embeddings-e5-large", CASCADE, "^"),
             ("каскад LoRA → судья", "cascade-lora-xlmr", CASCADE, "^"),
             ("LoRA", "lora-xlmr", LORA, "*"),
             ("e5-large + MLP", "embeddings-e5-large-mlp", EMB, "o")]
    free_x = 0.001
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    style(fig, ax)
    for label, name, color, marker in items:
        data = json.loads((RESULTS_DIR / name / "golden.json").read_text(encoding="utf-8"))
        cost = data["cost_per_1000_usd"] or 0.0
        x = cost if cost > 0 else free_x
        f1 = golden(name)["f1"]
        ax.scatter(x, f1, s=220 if marker == "*" else 90, color=color, marker=marker, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        dx, dy, ha = {"каскад e5-large → судья": (0, 10, "center"), "каскад e5-base → судья": (0, -12, "center")}.get(label, (10, 0, "left"))
        ax.annotate(label, (x, f1), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(free_x / 2, 1.0)
    ax.set_ylim(0.6, 1.04)
    # на логарифмической шкале нуля нет: бесплатные модели стоят у левого края, и эта отметка подписана «0»
    ax.set_xticks([free_x, 0.01, 0.1, 1.0])
    ax.set_xticklabels(["0", "0,01", "0,1", "1"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("цена за 1000 запросов, $", color=INK_2)
    ax.set_ylabel("F1 на golden", color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Качество и цена", loc="center", color=INK, fontsize=12)
    save(fig, out)


PR_OFFSET = {
    "e5-small + логрегрессия": (-8, 4, "right"),
    "e5-small + MLP": (8, 6, "left"),
    "e5-small + kNN": (0, -12, "center"),
    "e5-base + логрегрессия": (-8, 0, "right"),
    "e5-base + MLP": (8, -4, "left"),
    "e5-base + kNN": (-8, -4, "right"),
    "e5-large + логрегрессия": (-8, 0, "right"),
    "e5-large + MLP": (-8, -8, "right"),
    "e5-large + kNN": (8, -4, "left"),
    "LoRA": (10, 4, "left"),
    "каскад LoRA → судья": (-10, 6, "right"),
    "каскад e5-base → судья": (8, 0, "left"),
    "каскад e5-large → судья": (-8, 10, "right"),
    "судья Haiku 5.5": (-8, -4, "right"),
    "ключевые слова": (8, 0, "left"),
}


def pr_map(out: Path) -> None:
    """precision против recall на golden: PR-кривая для всех подходов без prompt tuning"""
    fig, ax = plt.subplots(figsize=(9, 6.2))
    style(fig, ax)
    rows = []
    for s, sn in SIZES:
        for c, cn in CLASSIFIERS:
            g = golden(f"embeddings-e5-{s}{c}")
            rows.append((f"{sn} + {cn}", g["recall"], g["precision"], EMB, "o", 90))
    lora = golden("lora-xlmr")
    rows.append(("LoRA", lora["recall"], lora["precision"], LORA, "*", 220))
    for name, label in [("cascade-lora-xlmr", "каскад LoRA → судья"),
                        ("cascade-embeddings-e5-base-mlp", "каскад e5-base → судья"),
                        ("cascade-embeddings-e5-large", "каскад e5-large → судья")]:
        g = golden(name)
        rows.append((label, g["recall"], g["precision"], CASCADE, "^", 90))
    judge = golden("judge-haiku")
    rows.append(("судья Haiku 5.5", judge["recall"], judge["precision"], JUDGE, "P", 90))
    kw = golden("keyword")
    rows.append(("ключевые слова", kw["recall"], kw["precision"], KEYWORD, "x", 90))
    ax.axhline(0.5, color=INK_2, linewidth=1, linestyle="--", zorder=1)
    ax.text(1.0, 0.5, " случайный выбор (баланс 1:1)", va="bottom", ha="right", fontsize=8.5, color=INK_2)
    for label, rec, prec, color, marker, size in rows:
        ax.scatter(rec, prec, s=size, color=color, marker=marker, edgecolor=SURFACE if marker != "x" else None,
                   linewidth=0.8 if marker != "x" else 1.5, zorder=3)
        dx, dy, ha = PR_OFFSET[label]
        ax.annotate(label, (rec, prec), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=8.5, color=INK)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("полнота (recall): какую долю атак нашли", color=INK_2)
    ax.set_ylabel("точность (precision): какая доля отмеченных атаками - атаки", color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Точность против полноты на golden", loc="left", color=INK, fontsize=12)
    save(fig, out)


CANDIDATES = [(10, 0, "left", "center"), (-10, 0, "right", "center"), (0, 11, "center", "bottom"), (0, -11, "center", "top"),
              (9, 9, "left", "bottom"), (-9, 9, "right", "bottom"), (9, -9, "left", "top"), (-9, -9, "right", "top"),
              (24, 0, "left", "center"), (-24, 0, "right", "center"), (0, 24, "center", "bottom"), (0, -24, "center", "top"),
              (22, 16, "left", "bottom"), (-22, 16, "right", "bottom"), (22, -16, "left", "top"), (-22, -16, "right", "top")]


def place_labels(fig, ax, pts, fontsize=8) -> None:
    """расставляет подписи у точек так, чтобы они не пересекались друг с другом и не закрывали маркеры"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marks = [ax.transData.transform((p["x"], p["y"])) for p in pts]
    boxes = []
    for p in sorted(pts, key=lambda q: -q["y"]):
        best = None
        for dx, dy, ha, va in CANDIDATES:
            a = ax.annotate(p["text"], (p["x"], p["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK, zorder=4)
            bb = a.get_window_extent(renderer).expanded(1.04, 1.25)
            bad = sum(bb.overlaps(o) for o in boxes) * 2 + sum(bb.contains(*m) for m in marks)
            if best is None or bad < best[0]:
                if best is not None:
                    best[1].remove()
                best = (bad, a, bb)
            else:
                a.remove()
            if bad == 0:
                break
        boxes.append(best[2])


def _all_points() -> list[dict]:
    """все одноступенчатые детекторы и все каскады на golden"""
    rows = []
    stages = [(f"{sn} + {cn}", f"embeddings-e5-{s}{c}") for s, sn in SIZES for c, cn in CLASSIFIERS] + [("LoRA", "lora-xlmr")]
    for label, fs in stages:
        g, cg = golden(fs), golden("cascade-" + fs)
        cdata = json.loads((RESULTS_DIR / ("cascade-" + fs) / "golden.json").read_text(encoding="utf-8"))
        rows.append({"label": label, "name": fs, "g": g, "c": cg, "cost": cdata["cost_per_1000_usd"] or 0.0})
    return rows


def _numbered(pairs: list[dict]) -> list[dict]:
    """нумерует пары «первая ступень - каскад» по убыванию F1 каскада"""
    out = sorted(pairs, key=lambda r: -r["c"]["f1"])
    for i, r in enumerate(out, 1):
        r["n"] = i
    return out


def _key(fig, pairs, x=0.735, y0=0.86, step=0.036) -> None:
    """ключ номеров справа от графика: номера выровнены по правому краю"""
    fig.text(x, y0 + 0.045, "Номер: подход", fontsize=11, color=INK, weight="bold")
    for k, r in enumerate(pairs):
        fig.text(x + 0.014, y0 - k * step, str(r["n"]), fontsize=10, color=INK, ha="right")
        fig.text(x + 0.024, y0 - k * step, r["label"], fontsize=10, color=INK)


def _marks(ax, x_single, y_single, x_casc, y_casc, lora):
    ax.scatter(x_single, y_single, s=200 if lora else 70, color=LORA if lora else EMB, marker="*" if lora else "o", edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.scatter(x_casc, y_casc, s=80, color=CASCADE, marker="^", edgecolor=SURFACE, linewidth=0.8, zorder=3)


def quality_map_all(out: Path) -> None:
    """карта качества: все одноступенчатые детекторы и все каскады, номера пар и ключ справа"""
    fig, ax = plt.subplots(figsize=(13, 7.4))
    style(fig, ax)
    fig.subplots_adjust(left=0.07, right=0.70)
    pairs = _numbered(_all_points())
    pts = []
    for r in pairs:
        g, c = r["g"], r["c"]
        _marks(ax, g["fpr"], g["recall"], c["fpr"], c["recall"], r["label"] == "LoRA")
        pts.append({"x": g["fpr"], "y": g["recall"], "text": str(r["n"])})
        pts.append({"x": c["fpr"], "y": c["recall"], "text": str(r["n"])})
    for label, name, color, marker in [("судья Haiku 5.5", "judge-haiku", JUDGE, "P"), ("ключевые слова", "keyword", KEYWORD, "x"),
                                       ("prompt tuning, 1", "prompt-tuning-xlmr/run1", PROMPT, "X"), ("prompt tuning, 2", "prompt-tuning-xlmr/run2", PROMPT, "X")]:
        g = golden(name)
        ax.scatter(g["fpr"], g["recall"], s=90, color=color, marker=marker, zorder=3, **({} if marker == "x" else {"edgecolor": SURFACE, "linewidth": 0.8}))
        manual = {"judge-haiku": (4, 14, "left", "bottom"), "prompt-tuning-xlmr/run1": (-10, 0, "right", "center"),
                  "prompt-tuning-xlmr/run2": (-8, 12, "right", "bottom")}
        if name in manual:
            dx, dy, ha, va = manual[name]
            ax.annotate(label, (g["fpr"], g["recall"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=9, color=INK)
        else:
            pts.append({"x": g["fpr"], "y": g["recall"], "text": label})
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("доля ложных срабатываний (FPR), левее лучше", color=INK_2)
    ax.set_ylabel("полнота (recall), выше лучше", color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Качество на golden: какие атаки ловим и сколько безопасных задеваем", loc="left", color=INK, fontsize=12)
    handles = [
        Line2D([], [], marker="o", color=EMB, linestyle="", label="первая ступень без судьи"),
        Line2D([], [], marker="*", color=LORA, linestyle="", markersize=12, label="LoRA без судьи"),
        Line2D([], [], marker="^", color=CASCADE, linestyle="", label="каскад с судьёй (тот же номер, что у первой ступени)"),
        Line2D([], [], marker="P", color=JUDGE, linestyle="", label="судья Haiku 5.5"),
        Line2D([], [], marker="X", color=PROMPT, linestyle="", label="prompt tuning (три запуска, не обучился)"),
        Line2D([], [], marker="x", color=KEYWORD, linestyle="", label="ключевые слова"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    place_labels(fig, ax, pts, fontsize=9)
    _key(fig, pairs)
    save(fig, out)


def cost_f1_all(out: Path) -> None:
    """качество против цены: все первые ступени стоят 0, каскады и судья правее; номера пар и ключ справа"""
    free_x = 0.001
    fig, ax = plt.subplots(figsize=(13, 7.4))
    style(fig, ax)
    fig.subplots_adjust(left=0.07, right=0.70)
    pairs = _numbered(_all_points())
    pts = []
    for r in pairs:
        cx = r["cost"] if r["cost"] > 0 else free_x
        _marks(ax, free_x, r["g"]["f1"], cx, r["c"]["f1"], r["label"] == "LoRA")
        pts.append({"x": free_x, "y": r["g"]["f1"], "text": str(r["n"])})
        pts.append({"x": cx, "y": r["c"]["f1"], "text": str(r["n"])})
    j = json.loads((RESULTS_DIR / "judge-haiku" / "golden.json").read_text(encoding="utf-8"))
    ax.scatter(j["cost_per_1000_usd"], golden("judge-haiku")["f1"], s=90, color=JUDGE, marker="P", edgecolor=SURFACE, linewidth=0.8, zorder=3)
    pts.append({"x": j["cost_per_1000_usd"], "y": golden("judge-haiku")["f1"], "text": "судья Haiku 5.5"})
    ax.set_xscale("log")
    ax.set_xlim(free_x / 2, 1.6)
    ax.set_ylim(0.6, 1.04)
    ax.set_xticks([free_x, 0.01, 0.1, 1.0])
    ax.set_xticklabels(["0", "0,01", "0,1", "1"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("цена за 1000 запросов, $", color=INK_2)
    ax.set_ylabel("F1 на golden", color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Качество и цена", loc="center", color=INK, fontsize=12)
    handles = [
        Line2D([], [], marker="o", color=EMB, linestyle="", label="первая ступень без судьи (0 $)"),
        Line2D([], [], marker="*", color=LORA, linestyle="", markersize=12, label="LoRA без судьи (0 $)"),
        Line2D([], [], marker="^", color=CASCADE, linestyle="", label="каскад с судьёй (тот же номер, что у первой ступени)"),
        Line2D([], [], marker="P", color=JUDGE, linestyle="", label="судья Haiku 5.5"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    place_labels(fig, ax, pts, fontsize=9)
    _key(fig, pairs)
    save(fig, out)


def f1_pairs(out: Path) -> None:
    """F1 каждой первой ступени без судьи и в каскаде с судьёй; справа доля запросов судье"""
    pairs = sorted(_all_points(), key=lambda r: r["c"]["f1"])  # снизу вверх: лучший каскад наверху
    fig, ax = plt.subplots(figsize=(11, 7.4))
    style(fig, ax)
    h = 0.36
    for i, r in enumerate(pairs):
        lora = r["label"] == "LoRA"
        share = json.loads((RESULTS_DIR / ("cascade-" + r["name"]) / "golden.json").read_text(encoding="utf-8"))["judged_share"]
        single, casc = r["g"]["f1"], r["c"]["f1"]
        ax.barh(i + h / 2, single, height=h, color=LORA if lora else EMB, alpha=0.55, zorder=3)
        ax.barh(i - h / 2, casc, height=h, color=CASCADE, zorder=3)
        ax.text(single + 0.008, i + h / 2, ru(single), va="center", fontsize=8.5, color=INK_2)
        ax.text(casc + 0.008, i - h / 2, ru(casc), va="center", fontsize=8.5, color=INK)
        ax.text(1.05, i, f"судье {share * 100:.1f}%".replace(".", ","), va="center", fontsize=9, color=INK_2)
    ax.axvline(1.0, color=JUDGE, linewidth=1.6, linestyle="--", zorder=2)
    ax.text(1.0, len(pairs) - 0.35, "судья один: 1,00", ha="center", va="bottom", fontsize=9, color=JUDGE)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([r["label"] for r in pairs], fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.7, len(pairs) + 0.1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0,2", "0,4", "0,6", "0,8", "1"])
    ax.set_xlabel("F1 на golden", color=INK_2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("F1: первая ступень и каскад с судьёй", loc="center", color=INK, fontsize=12)
    handles = [Patch(color=EMB, alpha=0.55, label="первая ступень без судьи"),
               Patch(color=LORA, alpha=0.55, label="LoRA без судьи"),
               Patch(color=CASCADE, label="каскад с судьёй")]
    ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, out)


def main() -> None:
    heatmap(OUT_DIR / "heatmap_e5")
    quality_map(OUT_DIR / "quality_map")
    pr_map(OUT_DIR / "pr_map")
    f1_bars(OUT_DIR / "f1_bars")
    benign_fpr(OUT_DIR / "benign_fpr")
    cost_f1(OUT_DIR / "cost_f1")
    quality_map_all(OUT_DIR / "quality_map_all")
    cost_f1_all(OUT_DIR / "cost_f1_all")
    f1_pairs(OUT_DIR / "f1_pairs")
    print(f"графики в {OUT_DIR}: heatmap_e5, quality_map, f1_bars, benign_fpr, cost_f1 (png и svg)")


if __name__ == "__main__":
    main()
