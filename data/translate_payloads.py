"""переводит часть английских фраз атак на русский локальной моделью (без обращений к API)

нужен, чтобы в русских данных были не только игровые фразы про пароль, но и другие типы атак
результат: data/raw/translated/en_ru_payloads.jsonl (в git не попадает), строки {"en": ..., "ru": ..., "category": ..., "source": ...}
использование: uv run python -m data.translate_payloads --n 700
"""
import argparse
import json
import random
from pathlib import Path

from data.build_splits import SEED, load_public

MODEL = "Helsinki-NLP/opus-mt-en-ru"
OUT = Path("data/raw/translated/en_ru_payloads.jsonl")


def main(n: int) -> None:
    from transformers import MarianMTModel, MarianTokenizer

    payloads, _ = load_public(include_translated=False)
    en = [p for p in payloads if p[1] == "en"]
    random.Random(SEED).shuffle(en)
    en = en[:n]
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # докачка: уже переведённые фразы не переводятся заново
    translated = {json.loads(line)["en"] for line in open(OUT, encoding="utf-8")} if OUT.exists() else set()
    en = [p for p in en if p[0] not in translated]
    print(f"уже переведено: {len(translated)}, осталось: {len(en)}", flush=True)
    done = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(0, len(en), 16):
            batch = en[i:i + 16]
            enc = tok([p[0] for p in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = model.generate(**enc, max_new_tokens=128, num_beams=2)
            for p, ru in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
                f.write(json.dumps({"en": p[0], "ru": ru, "category": p[2], "source": p[3]}, ensure_ascii=False) + "\n")
            f.flush()
            done += len(batch)
            print(f"переведено {done}/{len(en)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=700)
    main(parser.parse_args().n)
