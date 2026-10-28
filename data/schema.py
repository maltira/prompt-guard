"""формат записи данных и чтение/запись JSONL"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

LABELS = {"attack", "benign"}
LANGS = {"ru", "en"}
BORDERLINE = "benign_borderline"


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    label: str
    lang: str
    category: str
    source: str

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"{self.id}: label {self.label!r} не из {sorted(LABELS)}")
        if self.lang not in LANGS:
            raise ValueError(f"{self.id}: lang {self.lang!r} не из {sorted(LANGS)}")
        if not self.id or not self.text.strip():
            raise ValueError("id и text не могут быть пустыми")
        if self.category == BORDERLINE and self.label != "benign":
            raise ValueError(f"{self.id}: {BORDERLINE} допустима только для label=benign")


def read_jsonl(path: str | Path) -> list[Record]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(Record(**json.loads(line)))
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"повторяющиеся id: {dup[:5]}")
    return records


def write_jsonl(path: str | Path, records: list[Record]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
