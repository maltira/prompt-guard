"""сборка data/splits/{train,val,golden}.jsonl из открытых наборов, шаблонов и ручного golden

порядок: загрузка - фильтры по модели угроз - вставка фраз атак в письма - дедупликация -
исключение близких к golden - разбиение

фразы атак берутся из открытых наборов только если похожи на инструкцию ассистенту, и вставляются в
нейтральные письма; прямые просьбы, джейлбрейки и мусорные строки в атаки не попадают
"""
import argparse
import glob
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from data.build_golden import build as build_golden
from data.golden_hash import GoldenChangedError, register, sha256_bytes
from data.schema import Record, dumps_jsonl, write_jsonl
from data.synth import benign_with_markup, embed_text, generate, quoted_negative

RAW = Path("data/raw")
SPLITS = Path("data/splits")
SEED = 13
DEDUP_THRESHOLD = 0.92  # косинус tf-idf внутри train/val
GOLDEN_THRESHOLD = 0.60  # близость к golden, такие примеры уходят из train/val
SHINGLE_N, SHINGLE_MAX = 6, 2  # больше SHINGLE_MAX общих отрезков из SHINGLE_N слов с golden: убираем
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


# категории neuralchemy вне модели угроз (джейлбрейки, смена личности, шифрование, многоходовые, мусор)
OUT_OF_SCOPE_CATEGORIES = {"adversarial", "jailbreak", "encoding", "token_smuggling", "persona_replacement", "multi_turn",
                           "crescendo", "many_shot", "token_injection", "model_fingerprinting", "encoding_obfuscation",
                           "code_execution", "chain_of_thought", "control"}
# признаки инструкции, обращённой к ассистенту
CUE = re.compile(
    r"ignore|disregard|forget|override|bypass|reveal|disclose|leak|system prompt|previous instructions|"
    r"instructions|initial prompt|your prompt|password|secret|api key|confidential|hidden|from now on|new task|"
    r"instead of|do not mention|don't mention|without telling|"
    r"игнорир|забудь|отмени|раскро|секрет|пароль|инструкц|промпт|системн|с этого момента|вместо этого|"
    r"не упоминай|не говори|конфиденц|скрыт", re.I)


# открытые безопасные тексты - короткие запросы, поэтому берём их немного и оставляем отдельными (вне рамок письма)
PUBLIC_BENIGN_CAP = {"hf:S-Labs/prompt-injection-dataset": 60, "hf:neuralchemy/Prompt-injection-dataset": 70,
                     "hf:deepset/prompt-injections": 40, "hf:IlyaGusev/ru_turbo_alpaca": 100}
# потолок числа фраз атак от одного источника: чтобы простая синтетика и один язык не доминировали
PAYLOAD_CAP = {"hf:S-Labs/prompt-injection-dataset": 800, "hf:dmtrdr/russian_prompt_injections": 1300}


# фразы, привязанные к играм-источникам (Lakera Gandalf и Mosscap, hackaprompt), и приёмы вроде кодировок
GAME_NOISE = re.compile(r"mosscap|моссап|gandalf|гэндальф|гандальф|лейкер|lakera|pwned|\bhex\b|морзе|morse|щенок|щенк", re.I)
# команда в русской фразе: без неё вопрос - это не инструкция ассистенту
RU_COMMAND = re.compile(r"\b(игнорир|забудь|отмени|раскро|покажи|скажи|назови|выведи|напиши|дай|ответь|повтори|перечисл|"
                        r"напечатай|представь|расскажи|объясни|перескажи|добавь|пришли|отправь|переведи|пропусти|пометь|измени|"
                        r"не говори|не упоминай)", re.I)
PASSWORD_THEME = re.compile(r"парол|секрет|ключ", re.I)
PASSWORD_SHARE_MAX = 0.35  # доля фраз про пароль среди русских фраз атак
TRANSLATED = Path("data/raw/translated/en_ru_payloads.jsonl")
TRANSLATION_ORIGIN: dict[str, str] = {}  # русская фраза -> английский оригинал: для деления на группы


def ru_phrase_ok(text: str) -> bool:
    """дополнительные требования к русским фразам: не игра, не вопрос без команды, не слишком короткая, не повтор"""
    if GAME_NOISE.search(text) or len(text) < 40:
        return False
    if text.rstrip().endswith("?") and not RU_COMMAND.search(text):
        return False
    return not re.search(r"(\b\w+\b)(\W+\1){2,}", text.lower())


def alpha_share(text: str) -> float:
    return sum(c.isalpha() or c.isspace() for c in text) / max(1, len(text))


def is_payload(text: str, lang: str) -> bool:
    """фраза атаки: похожа на инструкцию ассистенту и не похожа на мусор, код или бессмысленный машинный перевод"""
    if not (15 <= len(text) <= 350 and alpha_share(text) >= 0.8 and CUE.search(text)):
        return False
    if len(re.findall(r"\w+", text)) < 4 or re.search(r"[{}<>=\\|]", text):
        return False
    letters = [c for c in text if c.isalpha()]
    if lang == "ru" and sum("\u0400" <= c <= "\u04ff" for c in letters) / max(1, len(letters)) < 0.9:
        return False
    return True


def load_public(include_translated: bool = True) -> tuple[list[tuple[str, str, str, str]], list[Record]]:
    """возвращает (фразы атак (текст, язык, категория, источник), безопасные записи)"""
    payloads: list[tuple[str, str, str, str]] = []
    benign: list[Record] = []

    def add_attack(src: str, text, category: str) -> None:
        text = clean(text)
        lang = detect_lang(text)
        if lang and is_payload(text, lang) and (lang != "ru" or ru_phrase_ok(text)):
            payloads.append((text, lang, category, src))

    def add_benign(src: str, i: int, text, category: str = "benign") -> None:
        rec = make(src, i, text, "benign", category)
        if rec and alpha_share(rec.text) >= 0.8:
            benign.append(rec)

    # deepset: небольшой, чистая разметка
    src = "hf:deepset/prompt-injections"
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RAW / "deepset__prompt-injections/*.parquet")))
    for i, (t_, y) in enumerate(zip(df["text"], df["label"])):
        add_attack(src, t_, "direct_injection") if y == 1 else add_benign(src, i, t_)
    # S-Labs: простая синтетика
    src = "hf:S-Labs/prompt-injection-dataset"
    df = pd.read_csv(RAW / "S-Labs__prompt-injection-dataset/data__train.csv").drop_duplicates("text")
    for i, (t_, y) in enumerate(zip(df["text"], df["label"])):
        if y == 1:
            add_attack(src, t_, "direct_injection")
        elif i % 12 == 0:  # безопасных коротких запросов нужно не больше, чем писем
            add_benign(src, i, t_)
    # neuralchemy core: оставляем категории внутри модели угроз
    src = "hf:neuralchemy/Prompt-injection-dataset"
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RAW / "neuralchemy__Prompt-injection-dataset/core__*.parquet")))
    df = df.drop_duplicates("text")
    for i, row in enumerate(df.itertuples()):
        if row.label == 1:
            if row.category not in OUT_OF_SCOPE_CATEGORIES:
                add_attack(src, row.text, row.category)
        elif row.category == "edge_case" or "hard_negative" in list(row.tags):
            add_benign(src, i, row.text, "benign_borderline")
        elif i % 5 == 0:
            add_benign(src, i, row.text)
    # русские атаки: только источники с реальными инъекциями (без джейлбрейк-набора)
    src = "hf:dmtrdr/russian_prompt_injections"
    ru = pd.DataFrame(json.load(open(RAW / "dmtrdr__russian_prompt_injections/dataset.json", encoding="utf-8")))
    ru = ru[ru["source"].isin(["Lakera/mosscap_prompt_injection", "hackaprompt/hackaprompt-dataset"])].drop_duplicates("prompt_ru")
    for t_ in ru["prompt_ru"]:
        add_attack(src, t_, "direct_injection")
    # русские безопасные запросы: разнообразие против шаблонности синтетики
    src = "hf:IlyaGusev/ru_turbo_alpaca"
    rows = [json.loads(line) for line in open(RAW / "IlyaGusev__ru_turbo_alpaca/ru_turbo_alpaca.jsonl", encoding="utf-8")]
    suspicious = re.compile(r"игнорир|забудь|пароль|системн|инструкци|промпт|секрет|взлом|jailbreak", re.I)
    texts = []
    for r in rows:
        t_ = (r["instruction"].strip() + ("\n" + r["input"].strip() if r["input"].strip() else "")).strip()
        if 20 <= len(t_) <= 500 and not suspicious.search(t_):
            texts.append(t_)
    random.Random(SEED).shuffle(texts)
    for i, t_ in enumerate(texts[:600]):
        add_benign(src, i, t_)
    rng = random.Random(SEED)
    capped_benign = []
    for src in dict.fromkeys(r.source for r in benign):
        items = [r for r in benign if r.source == src]
        rng.shuffle(items)
        capped_benign += items[:PUBLIC_BENIGN_CAP.get(src, len(items))]
    benign = capped_benign
    capped = []
    for src in dict.fromkeys(p_[3] for p_ in payloads):
        items = [p_ for p_ in payloads if p_[3] == src]
        rng.shuffle(items)
        capped += items[:PAYLOAD_CAP.get(src, len(items))]
    if include_translated:
        if not TRANSLATED.exists():
            raise SystemExit(f"нет {TRANSLATED}: сначала python -m data.translate_payloads (перевод английских фраз на русский)")
        for line in open(TRANSLATED, encoding="utf-8"):
            row = json.loads(line)
            ru = clean(row["ru"])
            if is_payload(ru, "ru") and ru_phrase_ok(ru):
                TRANSLATION_ORIGIN[ru] = row["en"]
                capped.append((ru, "ru", row["category"], "hf:mt-" + row["source"].split(":")[1]))
    # среди русских фраз не больше PASSWORD_SHARE_MAX про пароль: иначе детектор выучит «пароль = атака»
    ru_all = [p_ for p_ in capped if p_[1] == "ru"]
    themed = [p_ for p_ in ru_all if PASSWORD_THEME.search(p_[0])]
    other = len(ru_all) - len(themed)
    keep_themed = int(PASSWORD_SHARE_MAX / (1 - PASSWORD_SHARE_MAX) * other)
    drop = set(id(p_) for p_ in rng.sample(themed, max(0, len(themed) - keep_themed)))
    capped = [p_ for p_ in capped if id(p_) not in drop]
    return capped, benign


GROUPS: dict[str, str] = {}  # id записи -> группа (хэш фразы атаки), для деления без утечек


def embed_attacks(payloads: list[tuple[str, str, str, str]], share_embedded: float = 0.85) -> list[Record]:
    """большая часть фраз атак вставляется в письмо, остальные остаются короткими прямыми текстами"""
    rng = random.Random(SEED)
    out = []
    for text, lang, category, src in payloads:
        # половина фраз вставляется в два разных носителя: больше разнообразия письма вокруг одной и той же атаки
        copies = [True, True] if rng.random() < 0.5 and rng.random() < share_embedded else [rng.random() < share_embedded]
        for embedded in copies:
            full = embed_text(rng, lang, text) if embedded else text
            rec_src = src.replace("hf:", "hf-embedded:", 1) if embedded else src
            digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
            rec = Record(id=f"{src.split(':')[1].replace('/', '_')[:14]}-e-{digest}", text=full, label="attack",
                         lang=lang, category=category, source=rec_src)
            GROUPS[rec.id] = hashlib.sha1(TRANSLATION_ORIGIN.get(text, text).encode("utf-8")).hexdigest()[:12]
            out.append(rec)
    return out


def hard_negatives(payloads: list[tuple[str, str, str, str]], n_per_lang: int) -> list[Record]:
    """трудные безопасные тексты: те же фразы атак, но процитированные в предупреждениях и разборах"""
    rng = random.Random(SEED + 3)
    out = []
    for lang in ("en", "ru"):
        pool = [p_ for p_ in payloads if p_[1] == lang]
        for text, _, _, src in rng.sample(pool, min(n_per_lang, len(pool))):
            full = quoted_negative(rng, lang, text)
            digest = hashlib.sha1(full.encode("utf-8")).hexdigest()[:8]
            rec = Record(id=f"hard-{lang}-{digest}", text=full, label="benign", lang=lang,
                         category="benign_borderline", source=f"synthetic-quoted:{src.split(':')[1][:20]}")
            GROUPS[rec.id] = hashlib.sha1(TRANSLATION_ORIGIN.get(text, text).encode("utf-8")).hexdigest()[:12]
            out.append(rec)
    return out


def add_markup(records: list[Record], share: float) -> list[Record]:
    """часть шаблонных безопасных писем получает такую же разметку, как письма с атаками (чтобы разметка не выдавала атаку)"""
    rng = random.Random(SEED + 4)
    out = []
    for r in records:
        if r.label == "benign" and r.source == "synthetic:template" and rng.random() < share:
            text = benign_with_markup(rng, r.lang)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
            r = Record(id=f"syn-{r.lang}-b-m2-{digest}", text=text, label="benign", lang=r.lang,
                       category=r.category, source=r.source)
        out.append(r)
    return out


def markup_benign(n_per_lang: int) -> list[Record]:
    """безопасные письма с той же разметкой (комментарии, скрытые блоки), что и у атак"""
    rng = random.Random(SEED + 1)
    out = []
    for lang in ("en", "ru"):
        for _ in range(n_per_lang):
            text = benign_with_markup(rng, lang)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
            out.append(Record(id=f"syn-{lang}-b-m-{digest}", text=text, label="benign", lang=lang,
                              category="benign", source="synthetic:markup"))
    return out


def shingles(text: str, n: int = SHINGLE_N) -> set:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def drop_shingle_overlap(records: list[Record], golden: list[Record]) -> tuple[list[Record], int]:
    """убирает записи, делящие с golden несколько длинных словесных отрезков (почти копии его фраз)"""
    gold = set().union(*(shingles(g.text) for g in golden))
    keep = [r for r in records if len(shingles(r.text) & gold) <= SHINGLE_MAX]
    return keep, len(records) - len(keep)


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


def main(update_golden_hash: bool = False) -> None:
    golden = build_golden()
    # хэш проверяется до записи любых файлов: при расхождении golden.jsonl не трогаем
    golden_hash = sha256_bytes(dumps_jsonl(golden).encode("utf-8"))
    try:
        hash_status = register(golden_hash, update=update_golden_hash)
    except GoldenChangedError as e:
        raise SystemExit(f"остановлено: {e}")
    payloads, public_benign = load_public()
    print(f"фраз атак из открытых наборов после фильтра: {len(payloads)}", dict(Counter((p[1], p[3].split(':')[1][:12]) for p in payloads)))
    synthetic = add_markup(generate({"en": (200, 700, 250, 20), "ru": (380, 430, 250, 20)}), 0.55)
    pool = embed_attacks(payloads) + public_benign + markup_benign(500) + hard_negatives(payloads, 350) + synthetic
    print("до фильтров:", len(pool))
    pool, n_shingle = drop_shingle_overlap(pool, golden)
    print(f"убрано по общим отрезкам с golden: {n_shingle}")
    pool, dstats = dedup(pool, golden)
    dstats["shingle_overlap"] = n_shingle
    print("после дедупликации:", len(pool), dstats)
    strata = [f"{r.label}|{r.lang}|{r.source.split(':')[0]}" for r in pool]
    groups = [GROUPS.get(r.id, r.id) for r in pool]
    # делим группами: одна и та же фраза атаки (и её цитаты в безобидных текстах) целиком в train или в val
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(splitter.split(pool, strata, groups))
    train, val = [pool[i] for i in train_idx], [pool[i] for i in val_idx]
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
    (SPLITS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("train", "val", "golden"):
        s = stats[name]
        print(f"{name}: {s['total']} | {s['label']} | {s['lang']}")
    print(f"golden sha256 ({hash_status}):", golden_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-golden-hash", action="store_true",
                        help="записать новый хэш golden, если он намеренно изменился")
    main(parser.parse_args().update_golden_hash)
