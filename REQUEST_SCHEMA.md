# Схема обработки заметок

Этот документ описывает полный путь обработки заметки: от сырого Markdown-файла до готовых персональных карточек и связанных заметок.

## Оглавление

1. [Общая схема пайплайна](#общая-схема-пайплайна)
2. [Этап 1: Извлечение метаданных](#этап-1-извлечение-метаданных)
3. [Этап 2: Извлечение упоминаний (двухпроходная схема)](#этап-2-извлечение-упоминаний)
4. [Этап 3: Нормализация имён](#этап-3-нормализация-имён)
5. [Этап 4: Сканирование всех страниц](#этап-4-сканирование-всех-страниц)
6. [Этап 5: Кластеризация персон](#этап-5-кластеризация-персон)
7. [Этап 6: Генерация персональных заметок](#этап-6-генерация-персональных-заметок)
8. [Этап 7: Линковка заметок](#этап-7-линковка-заметок)
9. [Структуры данных](#структуры-данных)
10. [Файловая структура](#файловая-структура)

---

## Общая схема пайплайна

```
┌─────────────────┐
│  Markdown файл  │  data/pages/8331.md
│  (сырой текст)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ЭТАП 1          │
│ Метаданные      │  → NoteMetadata (год, место, тема)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ЭТАП 2          │
│ Извлечение      │  → MentionExtractionResponse (до 50 упоминаний)
│ упоминаний      │  → MentionGroupingResponse (группы = персоны)
│ (2 прохода LLM) │  → PersonExtractionResponse (до 20 персон)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ЭТАП 3          │
│ Нормализация    │  → PersonLocalNormalized (полные ФИО, блокирующие ключи)
│ имён            │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ ЭТАП 4: Сканирование всех страниц                               │
│ cache/persons_local_normalized.jsonl                            │
│ (каждая строка = PersonCandidate — один человек в одной заметке)│
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ ЭТАП 5          │
│ Кластеризация   │  → Embedding-индекс → Эвристики → LLM-мэтчинг
│ персон          │  → Union-Find → GlobalPerson[]
└────────┬────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│ ЭТАП 6          │          │ ЭТАП 7          │
│ Генерация       │          │ Линковка        │
│ персон.заметок  │          │ исходных        │
│                 │          │ заметок         │
└────────┬────────┘          └────────┬────────┘
         │                            │
         ▼                            ▼
  data/obsidian/              data/obsidian/
  persons/*.md                items/*.md
```

---

## Этап 1: Извлечение метаданных

**Файл:** `src/tools/note_metadata.py`

**Вход:** Сырой Markdown-файл

**Выход:** `NoteMetadata`

### Что происходит

1. Загружается Markdown-файл с помощью библиотеки `frontmatter`
2. Извлекается основной текст (игнорируя существующий frontmatter)
3. LLM получает текст и извлекает:
   - `primary_year` — основной год эпизода
   - `year_start` / `year_end` — диапазон периода
   - `location` — географическое место
   - `topic` — краткое описание (1-2 предложения)
   - `reliability` — уверенность (0-1)

### Валидация

- Год должен быть в диапазоне 1000-2100
- Если `year_start > year_end`, они меняются местами
- Если год не найден в тексте, `reliability` снижается

### Пример выхода

```python
NoteMetadata(
    primary_year=1930,
    year_start=1928,
    year_end=1932,
    location="Москва",
    topic="Работа Корина над фресками храма",
    reliability=0.85
)
```

---

## Этап 2: Извлечение упоминаний

**Файлы:**
- `src/tools/mention_extractor.py` (проход 1)
- `src/tools/mention_grouper.py` (проход 2)
- `src/tools/people_extractor.py` (оркестрация)

Используется **двухпроходная схема** для повышения качества.

### Проход 1: Извлечение упоминаний

**Вход:** Текст заметки + `NoteMetadata`

**Выход:** `MentionExtractionResponse` — список упоминаний (до 50)

Каждое упоминание (`PersonMention`) содержит:

| Поле | Описание |
|------|----------|
| `mention_id` | Уникальный ID (m1, m2, ...) |
| `text_span` | Точный текст из документа |
| `context_snippet` | 1-2 предложения контекста (уникальные!) |
| `likely_person` | Boolean — похоже на человека? |
| `inline_year` | Год рядом с упоминанием (если есть) |

**Правила:**
- Одно и то же имя в разных контекстах = разные упоминания (макс. 3)
- Одно имя в одном контексте = 1 упоминание
- Исключаются: абстрактные группы ("крестьяне"), организации, географические названия

### Пример упоминаний

```json
{
  "mentions": [
    {
      "mention_id": "m1",
      "text_span": "П.Д. Корин",
      "context_snippet": "В 1930 году П.Д. Корин начал работу над фресками.",
      "likely_person": true,
      "inline_year": 1930
    },
    {
      "mention_id": "m2",
      "text_span": "Павел Дмитриевич Корин",
      "context_snippet": "Павел Дмитриевич Корин был учеником Нестерова.",
      "likely_person": true,
      "inline_year": null
    },
    {
      "mention_id": "m3",
      "text_span": "Нестеров",
      "context_snippet": "Павел Дмитриевич Корин был учеником Нестерова.",
      "likely_person": true,
      "inline_year": null
    }
  ]
}
```

### Проход 2: Группировка упоминаний

**Вход:** Список упоминаний + исходный текст

**Выход:** `MentionGroupingResponse` — группы упоминаний

LLM объединяет упоминания, относящиеся к одному человеку:

```json
{
  "groups": [
    {
      "group_id": "g1",
      "mention_ids": ["m1", "m2"],
      "canonical_name": "Павел Дмитриевич Корин",
      "is_person": true,
      "confidence": 0.95
    },
    {
      "group_id": "g2",
      "mention_ids": ["m3"],
      "canonical_name": "Нестеров",
      "is_person": true,
      "confidence": 0.9
    }
  ]
}
```

### Результат этапа: PersonLocal

После группировки создаются объекты `PersonLocal`:

```python
PersonLocal(
    note_id="8331",
    local_person_id="p1",
    surface_forms=["П.Д. Корин", "Павел Дмитриевич Корин"],
    canonical_name_in_note="Павел Дмитриевич Корин",
    is_person=True,
    confidence=0.95,
    note_year_context=1930,
    note_year_source="inline",
    snippet_evidence=[...],
    role="художник",
    role_confidence=0.8
)
```

---

## Этап 3: Нормализация имён

**Файл:** `src/tools/name_normalizer.py`

**Вход:** `PersonExtractionResponse` + текст + метаданные

**Выход:** `PersonNormalizationResponse`

### Что происходит

1. Для каждого `PersonLocal` LLM пытается восстановить полное ФИО
2. Разбивает имя на части (фамилия, имя, отчество)
3. Создаёт нормализованные ключи для блокировки при матчинге
4. Строит связи между сокращёнными и полными формами

### Результат: PersonLocalNormalized

Расширяет `PersonLocal` полями:

| Поле | Пример |
|------|--------|
| `normalized_full_name` | "Павел Дмитриевич Корин" |
| `name_parts.last_name` | "Корин" |
| `name_parts.first_name` | "Павел" |
| `name_parts.patronymic` | "Дмитриевич" |
| `normalized_last_name` | "korin" (транслит, нижний регистр) |
| `first_initial` | "p" |
| `patronymic_initial` | "d" |
| `abbreviation_links` | [{from: "П.Д. Корин", to: "Павел Дмитриевич Корин", confidence: 0.95}] |

### Блокирующие ключи

Поля `normalized_last_name`, `first_initial`, `patronymic_initial` используются для быстрой фильтрации кандидатов на этапе кластеризации:
- Приводятся к нижнему регистру
- ё → е
- Старорусские буквы нормализуются

---

## Этап 4: Сканирование всех страниц

**Файл:** `src/tools/scan_people.py`

**CLI команда:** `scan_people`

**Вход:** Все файлы `data/pages/*.md`

**Выход:** `cache/persons_local_normalized.jsonl`

### Что происходит

1. Итерация по всем Markdown-файлам
2. Для каждого файла выполняются этапы 1-3
3. Результаты фильтруются:
   - Только `is_person=true`
   - Только `confidence >= 0.2`
4. Сохранение в JSONL-формате

### Формат кэша (PersonCandidate)

Каждая строка — один человек в одной заметке:

```json
{
  "candidate_id": "8332:p1",
  "note_id": "8332",
  "local_person_id": "p1",
  "normalized_full_name": "Павел Дмитриевич Корин",
  "canonical_name_in_note": "П.Д. Корин",
  "surface_forms": ["П.Д. Корин", "Павел Дмитриевич"],
  "name_parts": {
    "last_name": "Корин",
    "first_name": "Павел",
    "patronymic": "Дмитриевич"
  },
  "normalized_last_name": "korin",
  "first_initial": "p",
  "patronymic_initial": "d",
  "note_year_context": 1930,
  "person_confidence": 0.95,
  "confidence_bucket": "high",
  "role": "художник",
  "role_confidence": 0.8,
  "is_person": true,
  "snippet_preview": "В 1930 году П.Д. Корин начал..."
}
```

### Buckets уверенности

| Bucket | Диапазон confidence |
|--------|---------------------|
| `high` | ≥ 0.8 |
| `medium` | 0.6 — 0.8 |
| `low` | 0.4 — 0.6 |
| `very_low` | < 0.4 |

---

## Этап 5: Кластеризация персон

**Файлы:**
- `src/tools/cluster_people.py` (основная логика)
- `src/tools/person_matcher.py` (LLM-матчинг)
- `src/tools/embedding_index.py` (эмбеддинги)

**CLI команда:** `cluster`

**Вход:** `cache/persons_local_normalized.jsonl`

**Выход:** `cache/persons_clusters.json`

### Шаг 1: Embedding-индекс

Используется sentence-transformers для поиска похожих кандидатов:

```yaml
# config.yaml
embedding:
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  batch_size: 32
  top_k: 12
  index_type: hnswlib
```

Для каждого кандидата ищутся top-k похожих (по эмбеддингу имени).

### Шаг 2: Эвристическая фильтрация пар

Функция `should_consider_for_llm()` отсеивает пары:

1. **Фильтр по уверенности:** не матчим, если оба кандидата low/very_low
2. **Фильтр по году:** не матчим, если разница годов > `max_year_diff` (по умолчанию 25)
3. **Фильтр по схожести:** не матчим, если score < `min_pair_score` (по умолчанию 1.5)

### Шаг 3: "Дешёвые" решения

Функция `cheap_decision()` пытается принять решение без LLM:

**Возвращает `"same_person"` если:**
- Точное совпадение `normalized_full_name`
- Совпадение имени + фамилии
- Совпадение канонических имён + годы в пределах 30 лет

**Возвращает `"different_person"` если:**
- Разные фамилии
- Одинаковая фамилия, но разные имена/инициалы
- Конфликтующие отчества
- Разница годов > 60
- Конфликтующие роли при разных именах

**Возвращает `None` если неясно** → передаётся в LLM.

### Шаг 4: LLM-матчинг

Для неопределённых пар вызывается LLM:

```python
PersonMatchDecision(
    left_candidate_id="8331:p1",
    right_candidate_id="8332:p1",
    relation="same_person",  # или "different_person", "unknown"
    confidence=0.92
)
```

### Шаг 5: Union-Find кластеризация

Алгоритм DSU (Disjoint Set Union):
1. Каждый кандидат — отдельная компонента
2. Для каждого решения `"same_person"` объединяем компоненты
3. Результат: связные компоненты = кластеры

### Результат: GlobalPerson

```python
GlobalPerson(
    global_person_id="gp_0001",
    canonical_full_name="Павел Дмитриевич Корин",
    year_key="1892-1967",
    disambiguation_key="художник",
    members=["8331:p1", "8332:p1", "8340:p2"],
    episodes=[
        EpisodeRef(note_id="8331", year=1930),
        EpisodeRef(note_id="8332", year=1932),
        EpisodeRef(note_id="8340", year=1935)
    ]
)
```

---

## Этап 6: Генерация персональных заметок

**Файл:** `src/tools/person_note_generator.py`

**CLI команда:** `gen_person_notes`

**Вход:** `cache/persons_clusters.json`

**Выход:** `data/obsidian/persons/*.md`

### Формат файла

```markdown
---
type: person
full_name: Павел Дмитриевич Корин
global_person_id: gp_0001
episodes:
  - note_id: 8331
    year: 1930
  - note_id: 8332
    year: 1932
---

# Павел Дмитриевич Корин

## Эпизоды
- 8331 — [[See note]]
- 8332 — [[See note]]

## Краткое резюме
_TODO_

## Связанные люди
_TODO_
```

### Именование файлов

- Пробелы → подчёркивания
- Максимум 150 символов
- Если слишком длинное: обрезка + SHA1-хэш

---

## Этап 7: Линковка заметок

**Файл:** `src/tools/note_linker.py`

**CLI команда:** `link_persons`

**Вход:**
- Оригинальные заметки (`data/pages/`)
- Глобальные персоны (`cache/persons_clusters.json`)
- Кандидаты (`cache/persons_local_normalized.jsonl`)

**Выход:** `data/obsidian/items/*.md`

### Что происходит

1. Строится маппинг: `candidate_id` → имя файла персоны
2. Группировка кандидатов по заметкам
3. Поиск surface_forms в тексте
4. Создание wiki-ссылок: `[[Pavel_Dmitrievich_Korin]]`
5. Запись модифицированных заметок

### Обработка форм имён

- Линкуются только уникальные формы (принадлежащие одному человеку в заметке)
- Генерируются варианты многословных имён ("Иван Петрович" ↔ "Петрович Иван")
- Избегаются неоднозначные ссылки

---

## Структуры данных

### Иерархия моделей

```
Markdown (сырой текст)
    ↓
NoteMetadata
    ├── primary_year, year_start, year_end
    ├── location, topic
    └── reliability
    ↓
PersonMention (≤50 на заметку)
    ├── mention_id, text_span
    ├── context_snippet
    └── likely_person, inline_year
    ↓
MentionGroup
    ├── group_id, mention_ids[]
    ├── canonical_name
    └── is_person, confidence
    ↓
PersonLocal (≤20 на заметку)
    ├── note_id, local_person_id
    ├── surface_forms[], canonical_name_in_note
    ├── is_person, confidence
    ├── note_year_context, note_year_source
    ├── snippet_evidence[]
    └── role, role_confidence
    ↓
PersonLocalNormalized
    ├── (все поля PersonLocal)
    ├── normalized_full_name
    ├── name_parts (last, first, patronymic)
    ├── normalized_last_name, first_initial, patronymic_initial
    └── abbreviation_links[]
    ↓
PersonCandidate (кэш, JSONL)
    ├── candidate_id = "note_id:local_person_id"
    ├── (все поля PersonLocalNormalized)
    ├── confidence_bucket
    └── snippet_preview
    ↓
GlobalPerson (результат кластеризации)
    ├── global_person_id
    ├── canonical_full_name
    ├── year_key, disambiguation_key
    ├── members[] (candidate_id)
    └── episodes[] (EpisodeRef)
```

### Ключевые модели (src/models.py)

| Модель | Назначение |
|--------|------------|
| `NoteMetadata` | Метаданные заметки (год, место, тема) |
| `PersonMention` | Одно упоминание имени в тексте |
| `MentionGroup` | Группа упоминаний одного человека |
| `PersonLocal` | Человек в контексте одной заметки |
| `PersonLocalNormalized` | + нормализованное ФИО и ключи |
| `PersonCandidate` | Глобальный кандидат для матчинга |
| `GlobalPerson` | Объединённый человек после кластеризации |
| `PersonMatchDecision` | Решение о совпадении двух кандидатов |

---

## Файловая структура

```
data/
├── pages/                          # Вход: сырые Markdown-заметки
│   ├── 8331.md
│   ├── 8332.md
│   └── ...
│
├── obsidian/
│   ├── persons/                    # Выход: персональные карточки
│   │   ├── Pavel_Dmitrievich_Korin.md
│   │   └── ...
│   └── items/                      # Выход: заметки со ссылками
│       ├── 8331.md
│       └── ...

cache/
├── persons_local_normalized.jsonl  # Промежуточный: все кандидаты
├── persons_clusters.json           # Финальный: результат кластеризации
└── embeddings/                     # Индекс эмбеддингов
    ├── embeddings.npy
    ├── index.hnsw
    └── ...

log_tracing/                        # Отладка (при SGR_TRACE_NOTES)
└── people_pipeline/
    └── 8331/
        ├── mentions_*.json
        ├── grouping_*.json
        └── ...
```

---

## Конфигурация

**Файл:** `config.yaml`

```yaml
matching:
  max_year_diff: 25        # Макс. разница годов для матчинга
  min_pair_score: 1.5      # Мин. эвристический score для LLM

embedding:
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  batch_size: 32
  top_k: 12                # Сколько похожих кандидатов искать
  index_type: hnswlib      # hnswlib | annoy | faiss
```

**Переменные окружения (.env):**

| Переменная | Описание |
|------------|----------|
| `OPENAI_BASE_URL` | URL LLM-сервера |
| `OPENAI_API_KEY` | API-ключ |
| `MODEL_NAME` | Название модели |
| `SGR_TRACE_NOTES` | ID заметок для трейсинга (через запятую) |

---

## CLI-команды

| Команда | Описание | Этапы |
|---------|----------|-------|
| `note_meta <path>` | Метаданные одной заметки | 1 |
| `note_people <path>` | Сырые персоны из заметки | 1-2 |
| `note_people_normalized <path>` | Нормализованные персоны | 1-3 |
| `scan_people` | Обработка всех заметок → кэш | 1-4 |
| `show_candidates` | Показать кандидатов из кэша | — |
| `cluster` | Кластеризация персон | 5 |
| `gen_person_notes` | Генерация персональных заметок | 6 |
| `link_persons` | Линковка исходных заметок | 7 |

---

## Точки вызова LLM

1. **Извлечение метаданных** — 1 вызов на заметку
2. **Извлечение упоминаний** — 1 вызов на заметку
3. **Группировка упоминаний** — 1 вызов на заметку
4. **Нормализация имён** — 1 вызов на заметку
5. **Матчинг персон** — 1 вызов на пару кандидатов (после фильтрации)

**Итого на одну заметку:** 4 вызова LLM (этапы 1-3)
**На кластеризацию:** O(n²) в худшем случае, на практике значительно меньше благодаря эмбеддингам и эвристикам.
