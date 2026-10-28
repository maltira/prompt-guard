# Данные

Выборки для обучения и оценки детекторов prompt injection.

## Выборки

| Файл | Размер | Для чего |
|---|---|---|
| `splits/train.jsonl` | 9 180 | обучение детекторов |
| `splits/val.jsonl` | 1 621 | подбор порогов и гиперпараметров, ранняя остановка |
| `splits/golden.jsonl` | 450 | итоговая оценка, только для измерений |
| `splits/stats.json` | | статистика по классам, языкам, категориям и источникам |
| `splits/golden.sha256` | | SHA-256 файла `golden.jsonl`, единственное место, где хранится хэш |

**Golden не используется ни для обучения, ни для подбора настроек.** Файл `splits/golden.jsonl` зафиксирован и больше не меняется.

**Хэш golden** хранится в одном месте - [splits/golden.sha256](splits/golden.sha256). Он нужен, чтобы заметить, что golden изменился: если хэш другой, прежние оценки сравнивать с новыми нельзя.

- Читают его все: `eval/run_eval.py` перед оценкой на golden сверяет файл с хэшем и останавливается при расхождении. Вручную: `sha256sum data/splits/golden.jsonl` и сравнить с содержимым `golden.sha256`.
- Пишет его только [build_splits.py](build_splits.py), и по правилам: нет файла - создаёт; хэш тот же - ничего не делает; хэш другой - **останавливается, ничего не перезаписывая**. Обновить хэш можно только явным флагом `python -m data.build_splits --update-golden-hash`, когда вы сознательно меняете golden.
- Логика хранения и проверки - в [golden_hash.py](golden_hash.py).

### Golden (450)

- 225 атак и 225 безопасных; 200 английских и 250 русских.
- Безопасные включают 92 пограничных (`benign_borderline`): тексты с лексикой атаки или признаками мошенничества, но безопасные.
- Все тексты написаны вручную. Русские: 175 адаптированы с английских сценариев (`custom:ru_adapted`), 75 придуманы с нуля (`custom:ru_native`). Метрики для этих срезов стоит считать отдельно.
- Все тексты считаются недоверенным содержимым (письмо, документ, страница), которое обрабатывает ассистент. Инструкция внутри такого текста - внедрение. Запрос самого пользователя в golden не входит.
- Категория атаки - основной приём, а не единственный: категории не исключают друг друга (например, смена роли может служить для выманивания секрета). Метрики по категориям читать с оговоркой.

### Train и val

Собираются из открытых наборов и шаблонной синтетики, делятся 85/15 с сохранением пропорций по классу, языку и источнику.

## Формат записи

Одна запись на строку JSONL, поля описаны в [schema.py](schema.py):

| Поле | Значения |
|---|---|
| `id` | уникальный идентификатор |
| `text` | текст письма, документа или запроса |
| `label` | `attack` или `benign` |
| `lang` | `ru` или `en` |
| `category` | атаки: `instruction_override`, `secret_exfiltration`, `fake_system_message`, `role_hijack`, `hidden_instruction`, `output_manipulation`, `direct_injection` и категории neuralchemy; безопасные: `benign`, `benign_borderline` |
| `source` | происхождение, см. ниже |

Значения `source`: `hf:<набор>` (открытый набор), `synthetic:template` и `synthetic:template_short` (шаблоны), `custom:handwritten` (английский golden), `custom:ru_adapted`, `custom:ru_native`.

## Устройство папки

```
data/
  schema.py        формат записи, чтение и запись JSONL
  synth.py         генератор шаблонных примеров для train и val
  build_golden.py  собирает golden из ручных файлов
  build_splits.py  собирает train, val, golden и stats.json
  custom/          ручные тексты golden: gold_{en,ru}_{attack,benign}.py
  raw/             скачанные открытые наборы (в git не попадают)
  splits/          результат сборки
```

В `custom/gold_*.py` тексты лежат списками пар `(категория, текст)`. В русских файлах два списка: `ITEMS` (адаптированные) и `NATIVE` (придуманные с нуля).

## Как собираются train и val

`python -m data.build_splits` делает по шагам:

1. Загружает открытые наборы из `raw/` (таблица ниже) и генерирует шаблонные примеры (`synth.py`).
2. Определяет язык, отбрасывает тексты не на русском и английском и слишком короткие.
3. Убирает точные дубликаты и почти-дубликаты внутри выборки (косинус tf-idf по символьным n-граммам от 0,92).
4. Убирает примеры, близкие к golden (косинус от 0,80), чтобы не было утечки.
5. Делит на train и val, проверяет, что id не пересекаются между train, val и golden.
6. Пишет `splits/*.jsonl` и `stats.json`.

Сборка детерминирована (`SEED = 13`). Golden каждый раз заново собирается из `custom/gold_*.py`. Если после правки этих файлов хэш изменился, сборка остановится, пока вы не передадите `--update-golden-hash`.

### Открытые наборы

| Набор | Лицензия | Что берём |
|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | все примеры, английские |
| [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) | MIT | по 1 200 на класс из train |
| [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) | Apache-2.0 | конфигурация `core`, по 1 200 на класс |
| [dmtrdr/russian_prompt_injections](https://huggingface.co/datasets/dmtrdr/russian_prompt_injections) | Apache-2.0 | до 1 200 русских атак из источников Lakera mosscap, hackaprompt, jackhhao |
| [IlyaGusev/ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca) | CC-BY-4.0 | 900 русских безопасных запросов |

Русские атаки из `dmtrdr` - машинный перевод английских наборов, а сам набор собран из источников с разными лицензиями. В golden они не используются.

Шаблонная синтетика: английских (атаки/безопасные/пограничные/короткие) 900/600/400/30, русских 900/1 100/550/30.

### Как восстановить `raw/`

Файлы скачиваются вручную с Hugging Face, имя локального файла - путь в репозитории, где `/` заменён на `__`:

- `raw/deepset__prompt-injections/*.parquet` (из `data/`)
- `raw/S-Labs__prompt-injection-dataset/data__train.csv`
- `raw/neuralchemy__Prompt-injection-dataset/core__{train,validation,test}-00000-of-00001.parquet`
- `raw/dmtrdr__russian_prompt_injections/dataset.json`
- `raw/IlyaGusev__ru_turbo_alpaca/ru_turbo_alpaca.jsonl` (распаковать `ru_turbo_alpaca.jsonl.zst` командой `zstd -d`)

## Известные ограничения

- Golden небольшой: погрешность метрик по срезам заметная (по паре «класс × язык» около ±6-7 п.п.), по русским «с нуля» (25 атак и 50 безопасных) выше.
- Русские адаптированные примеры повторяют сценарии английских, поэтому метрики по русскому - не полностью независимый эксперимент.
- Атаки и их безопасные «чистые версии» в golden написаны одним автором и очень похожи по носителю; это сложный тест, но он может отличаться от реального распределения.
- Синтетика в train однообразнее настоящих писем: годится для обучения, но не для оценки.
- На val простые детекторы показывают качество заметно выше, чем на golden. Итоговые цифры брать только с golden.
