"""сборка golden set из ручных файлов data/custom/gold_*.py"""
import hashlib
from collections import Counter

from data.custom import gold_en_attack, gold_en_benign, gold_ru_attack, gold_ru_benign
from data.schema import Record

# определения для golden
# - все тексты считаются недоверенным содержимым (письмо, документ, страница), которое обрабатывает ассистент
#   инструкция, адресованная ассистенту внутри такого текста, и есть внедрение; запрос самого пользователя
#   в golden не входит
# - category - основной приём атаки; категории не исключают друг друга (например, смена роли
#   может служить для выманивания секрета), поэтому метрики по категориям читать с оговоркой
# - benign_borderline - безопасный текст с лексикой атаки или признаками мошенничества
# источники: (язык, метка, список пар (категория, текст), происхождение)
# ru_adapted - сценарий параллелен английскому, ru_native - придуман с нуля
SOURCES = [
    ("en", "attack", gold_en_attack.ITEMS, "custom:handwritten"),
    ("ru", "attack", gold_ru_attack.ITEMS, "custom:ru_adapted"),
    ("en", "benign", gold_en_benign.ITEMS, "custom:handwritten"),
    ("ru", "benign", gold_ru_benign.ITEMS, "custom:ru_adapted"),
    ("ru", "attack", gold_ru_attack.NATIVE, "custom:ru_native"),
    ("ru", "benign", gold_ru_benign.NATIVE, "custom:ru_native"),
]


def make_id(lang: str, label: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"gold-{lang}-{label[0]}-{digest}"


def build() -> list[Record]:
    records = []
    for lang, label, items, source in SOURCES:
        for category, text in items:
            records.append(Record(
                id=make_id(lang, label, text), text=text, label=label, lang=lang,
                category=category, source=source))
    return records


def report(records: list[Record]) -> None:
    print("всего:", len(records))
    print("по метке и языку:", dict(Counter((r.label, r.lang) for r in records)))
    for (label, lang), n in sorted(Counter((r.label, r.lang) for r in records).items()):
        cats = Counter(r.category for r in records if r.label == label and r.lang == lang)
        print(f"  {label}/{lang}: {n}", dict(cats))
    print("по происхождению:", dict(Counter((r.lang, r.source) for r in records)))


if __name__ == "__main__":
    report(build())
