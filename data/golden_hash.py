"""единое хранилище хэша golden: читают все, пишет только build_splits"""
import hashlib
from pathlib import Path

HASH_FILE = Path("data/splits/golden.sha256")
GOLDEN_FILE = Path("data/splits/golden.jsonl")


class GoldenChangedError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_hash(path: Path = HASH_FILE) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def register(new_hash: str, path: Path = HASH_FILE, update: bool = False) -> str:
    """фиксирует хэш golden, вызывается только из build_splits.

    возвращает created, unchanged или updated. если хэш уже записан и отличается,
    без update=True бросает GoldenChangedError и ничего не меняет
    """
    stored = read_hash(path)
    if stored is None:
        path.write_text(new_hash + "\n", encoding="utf-8")
        return "created"
    if stored == new_hash:
        return "unchanged"
    if not update:
        raise GoldenChangedError(
            f"golden изменился: записан {stored}, получился {new_hash}. "
            "если это сделано намеренно, запустите с --update-golden-hash")
    path.write_text(new_hash + "\n", encoding="utf-8")
    return "updated"


def verify_file(golden_path: Path = GOLDEN_FILE, path: Path = HASH_FILE) -> str:
    """сверяет файл golden с записанным хэшем, возвращает хэш"""
    stored = read_hash(path)
    if stored is None:
        raise GoldenChangedError(f"не найден {path}, сначала запустите python -m data.build_splits")
    actual = sha256_bytes(golden_path.read_bytes())
    if actual != stored:
        raise GoldenChangedError(f"хэш {golden_path} не совпадает с записанным в {path}: {actual}")
    return actual
