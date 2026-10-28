"""сборка data/splits/{train,val,golden}.jsonl из открытых наборов, шаблонов и ручного golden

порядок: загрузка - фильтры - дедупликация - исключение близких к golden - разбиение
"""
import glob
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from data.build_golden import build as build_golden
from data.schema import Record, write_jsonl
from data.synth import generate

RAW = Path("data/raw")
SPLITS = Path("data/splits")
SEED = 13
DEDUP_THRESHOLD = 0.92  # косинус tf-idf внутри train/val
GOLDEN_THRESHOLD = 0.80  # близость к golden, такие примеры уходят из train/val
EN_STOP = {"the", "is", "are", "of", "and", "to", "in", "you", "what", "how", "your", "for", "with", "that", "this", "i", "a", "do", "can", "it"}
DE_STOP = {"der", "die", "das", "und", "ist", "nicht", "ich", "sie", "ein", "eine", "wie", "was", "zu", "mit", "für", "auf", "den", "von"}


def detect_lang(text: str) -> str | None:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    if sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters) > 0.5:
        return "ru"
    words = re.findall(r"[a-zA-Z]+", text.lower())
    en, de = sum(w in EN_STOP for w in words), sum(w in DE_STOP for w in words)
    return "en" if en > de and en >= 1 else None


def clean(text) -> str:
    return re.sub(r"\s+\n", "\n", str(text)).strip()


def make(src: str, i: int, text: str, label: str, category: str) -> Record | None:
    text = clean(text)
    lang = detect_lang(text)
    if lang is None or len(text) < 8:
        return None
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return Record(id=f"{src.split(':')[1].replace('/', '_')[:14]}-{digest}", text=text,
                  label=label, lang=lang, category=category, source=src)


def sample_balanced(df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    parts = [g.sample(min(len(g), n_per_class), random_state=SEED) for _, g in df.groupby("label")]
    return pd.concat(parts)


def load_public() -> list[Record]:
    out: list[Record] = []
    # deepset: небольшой, чистая разметка
    src = "hf:deepset/prompt-injections"
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RAW / "deepset__prompt-injections/*.parquet")))
    for i, (t, y) in enumerate(zip(df["text"], df["label"])):
        out.append(make(src, i, t, "attack" if y == 1 else "benign", "direct_injection" if y == 1 else "benign"))
    # S-Labs: синтетика, берем сбалансированную часть
    src = "hf:S-Labs/prompt-injection-dataset"
    df = pd.read_csv(RAW / "S-Labs__prompt-injection-dataset/data__train.csv").drop_duplicates("text")
    for i, (t, y) in enumerate(zip(*[sample_balanced(df, 1200)[c] for c in ("text", "label")])):
        out.append(make(src, i, t, "attack" if y == 1 else "benign", "direct_injection" if y == 1 else "benign"))
    # neuralchemy core: есть категории и сложные негативы
    src = "hf:neuralchemy/Prompt-injection-dataset"
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RAW / "neuralchemy__Prompt-injection-dataset/core__*.parquet")))
    df = sample_balanced(df.drop_duplicates("text"), 1200)
    for i, row in enumerate(df.itertuples()):
        hard = row.label == 0 and (row.category == "edge_case" or "hard_negative" in list(row.tags))
        if row.label == 1:
            cat = row.category
        else:
            cat = "benign_borderline" if hard else "benign"
        out.append(make(src, i, row.text, "attack" if row.label == 1 else "benign", cat))
    # русские атаки: только источники, где реально есть инъекции, и без мусора
    src = "hf:dmtrdr/russian_prompt_injections"
    ru = pd.DataFrame(json.load(open(RAW / "dmtrdr__russian_prompt_injections/dataset.json", encoding="utf-8")))
    ru = ru[ru["source"].isin(["Lakera/mosscap_prompt_injection", "hackaprompt/hackaprompt-dataset", "jackhhao/jailbreak-classification"])]
    ru = ru[ru["prompt_ru"].str.len().between(25, 600)].drop_duplicates("prompt_ru")
    ru = ru.sample(min(len(ru), 1200), random_state=SEED)
    for i, t in enumerate(ru["prompt_ru"]):
        out.append(make(src, i, t, "attack", "direct_injection"))
    # русские безопасные запросы: разнообразие против шаблонности синтетики
    src = "hf:IlyaGusev/ru_turbo_alpaca"
    rows = [json.loads(line) for line in open(RAW / "IlyaGusev__ru_turbo_alpaca/ru_turbo_alpaca.jsonl", encoding="utf-8")]
    suspicious = re.compile(r"игнорир|забудь|пароль|системн|инструкци|промпт|секрет|взлом|jailbreak", re.I)
    texts = []
    for r in rows:
        t = (r["instruction"].strip() + ("\n" + r["input"].strip() if r["input"].strip() else "")).strip()
        if 20 <= len(t) <= 500 and not suspicious.search(t):
            texts.append(t)
    random.Random(SEED).shuffle(texts)
    for i, t in enumerate(texts[:900]):
        out.append(make(src, i, t, "benign", "benign"))
    return [r for r in out if r is not None]


def dedup(records: list[Record], golden: list[Record]) -> tuple[list[Record], dict]:
    """убирает точные и почти-дубликаты, а также примеры, близкие к golden"""
    stats = Counter()
    seen, uniq = set(), []
    for r in records:
        key = re.sub(r"\s+", " ", r.text.lower())
        if key in seen:
            stats["exact_dup"] += 1
            continue
        seen.add(key)
        uniq.append(r)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    vec.fit([r.text for r in uniq + golden])
    X, G = vec.transform([r.text for r in uniq]), vec.transform([r.text for r in golden])
    near_gold = set()
    for start in range(0, X.shape[0], 1000):
        sim = (X[start:start + 1000] @ G.T).toarray()
        near_gold.update(start + int(i) for i in (sim.max(axis=1) >= GOLDEN_THRESHOLD).nonzero()[0])
    keep, kept_rows = [], []
    for i, r in enumerate(uniq):
        if i in near_gold:
            stats["near_golden"] += 1
            continue
        if kept_rows:
            sim = (X[i] @ X[kept_rows].T).toarray().max()
            if sim >= DEDUP_THRESHOLD:
                stats["near_dup"] += 1
                continue
        keep.append(r)
        kept_rows.append(i)
    return keep, dict(stats)


def main() -> None:
    golden = build_golden()
    pool = load_public() + generate({"en": (900, 600, 400, 30), "ru": (900, 1100, 550, 30)})
    print("до фильтров:", len(pool))
    pool, dstats = dedup(pool, golden)
    print("после дедупликации:", len(pool), dstats)
    strata = [f"{r.label}|{r.lang}|{r.source.split(':')[0]}" for r in pool]
    train, val = train_test_split(pool, test_size=0.15, random_state=SEED, stratify=strata)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("val", val), ("golden", golden)):
        write_jsonl(SPLITS / f"{name}.jsonl", recs)
    ids = [set(r.id for r in s) for s in (train, val, golden)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "пересечение id между выборками"
    stats = {}
    for name, recs in (("train", train), ("val", val), ("golden", golden)):
        stats[name] = {"total": len(recs),
                       "label": dict(Counter(r.label for r in recs)),
                       "lang": dict(Counter(r.lang for r in recs)),
                       "label_lang": {f"{k[0]}/{k[1]}": v for k, v in Counter((r.label, r.lang) for r in recs).items()},
                       "category": dict(Counter(r.category for r in recs).most_common()),
                       "source": dict(Counter(r.source for r in recs))}
    stats["dedup"] = dstats
    stats["golden_sha256"] = hashlib.sha256((SPLITS / "golden.jsonl").read_bytes()).hexdigest()
    (SPLITS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("train", "val", "golden"):
        s = stats[name]
        print(f"{name}: {s['total']} | {s['label']} | {s['lang']}")
    print("golden sha256:", stats["golden_sha256"])


if __name__ == "__main__":
    main()
