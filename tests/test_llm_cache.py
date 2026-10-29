from types import SimpleNamespace

from detectors.llm_cache import CachedLLM


class FakeClient:
    def __init__(self):
        self.n = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.n += 1
        msg = SimpleNamespace(content=f"ответ{self.n}")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_cache_hit(tmp_path):
    fake = FakeClient()
    llm = CachedLLM(model="m", cache_dir=tmp_path, client=fake)
    assert llm.complete("q") == llm.complete("q") == "ответ1"
    assert fake.n == 1


def test_key_depends_on_inputs(tmp_path):
    fake = FakeClient()
    llm = CachedLLM(model="m", cache_dir=tmp_path, client=fake)
    llm.complete("q")
    llm.complete("q", system="s")
    llm.complete("q", temperature=0.5)
    assert fake.n == 3
    # другая модель - другой ключ, общий каталог
    CachedLLM(model="m2", cache_dir=tmp_path, client=fake).complete("q")
    assert fake.n == 4


def test_min_interval_spaces_real_calls_but_not_cache_hits(tmp_path):
    import time
    fake = FakeClient()
    llm = CachedLLM(model="m", cache_dir=tmp_path, client=fake, min_interval_s=0.2)
    t0 = time.monotonic()
    llm.complete("a")
    llm.complete("b")
    assert time.monotonic() - t0 >= 0.2
    t1 = time.monotonic()
    llm.complete("a")  # из кэша, без паузы
    assert time.monotonic() - t1 < 0.1
