# План улучшения Grouper: алгоритмическая генерация кандидатов

## Проблема

Текущий grouper полностью полагается на LLM для группировки mentions. При большом количестве упоминаний (10+) LLM:
- Теряет часть mentions (возвращает 1 группу вместо 10)
- Объединяет разных людей в одну группу
- Работает нестабильно

## Решение

Разделить задачу на две части:
1. **Алгоритмическая генерация кандидатов** — детерминированно находим пары mentions, которые МОГУТ быть одним человеком
2. **LLM валидация** — упрощённый промпт только подтверждает/отклоняет предложенные пары

---

## Этап 1: Алгоритмическая генерация кандидатов

### 1.1 Извлечение компонентов имени

Для каждого mention извлекаем:
- `last_name_stem` — основа фамилии (без падежного окончания)
- `first_name` или `first_initial` — имя или инициал
- `patronymic` или `patronymic_initial` — отчество или инициал

```python
@dataclass
class ParsedMention:
    mention_id: str
    text_span: str
    last_name_stem: str | None      # "нижинск", "чаплин", "ларионов"
    first_name: str | None          # "Вацлав", "Чарли", None
    first_initial: str | None       # "В", "Ч", "М"
    patronymic: str | None          # "Фёдорович", None
    patronymic_initial: str | None  # "Ф", None
    is_initials_only: bool          # True для "М.Ф. Ларионова"
```

### 1.2 Парсинг имён

```python
def parse_mention(mention: PersonMention) -> ParsedMention:
    """
    Парсит text_span и извлекает компоненты имени.

    Примеры:
    - "Вацлав Нижинский" → last="нижинск", first="Вацлав", init="В"
    - "Нижинского" → last="нижинск", first=None
    - "М.Ф. Ларионова" → last="ларионов", first_init="М", patr_init="Ф"
    - "Чарли Чаплином" → last="чаплин", first="Чарли"
    - "Альфонсо XIII" → last=None (особый случай: монарх с номером)
    """
```

**Алгоритм парсинга:**

1. Токенизация: разбиваем на слова
2. Определение инициалов: паттерн `[А-ЯЁ]\.` или `[А-ЯЁ]\.[А-ЯЁ]\.`
3. Определение фамилии: обычно последнее слово (не инициал)
4. Извлечение основы: удаляем падежные окончания

```python
# Примеры парсинга
"Вацлав Нижинский"    → ["Вацлав", "Нижинский"] → last="Нижинский", first="Вацлав"
"Нижинского"          → ["Нижинского"] → last="Нижинского" → stem="Нижинск"
"М.Ф. Ларионова"      → ["М.", "Ф.", "Ларионова"] → last="Ларионова", inits=["М", "Ф"]
"Чарли Чаплином"      → ["Чарли", "Чаплином"] → last="Чаплином" → stem="Чаплин"
"патриарха Тихона"    → ["патриарха", "Тихона"] → title="патриарх", last="Тихона"
```

### 1.3 Генерация кандидатов на объединение

```python
def generate_merge_candidates(
    parsed_mentions: list[ParsedMention]
) -> list[MergeCandidate]:
    """
    Генерирует пары mentions, которые МОГУТ быть одним человеком.

    Правила генерации кандидатов:
    1. Совпадение основы фамилии (обязательно)
    2. Непротиворечивые имена/инициалы
    """
```

**Правила совпадения:**

| Mention A | Mention B | Кандидат? | Причина |
|-----------|-----------|-----------|---------|
| "Нижинск*" | "Нижинск*" | Да | Одна фамилия |
| "В. Нижинск*" | "Вацлав Нижинск*" | Да | Инициал совпадает |
| "И. Нижинск*" | "Вацлав Нижинск*" | Нет | Инициалы разные |
| "Чаплин*" | "Нижинск*" | Нет | Разные фамилии |
| "М.Ф. Ларионов*" | "Ларионов*" | Да | Фамилия совпадает |
| "М.Ф. Ларионов*" | "И.И. Ларионов*" | Нет | Инициалы разные |

```python
@dataclass
class MergeCandidate:
    mention_id_a: str
    mention_id_b: str
    confidence: float  # Алгоритмическая уверенность
    reason: str        # Почему считаем кандидатами
```

### 1.4 Пример генерации кандидатов

**Входные mentions (157163):**
```
m1: "Дягилева"           → stem="дягилев"
m2: "Вацлав Нижинский"   → stem="нижинск", first="Вацлав"
m3: "Нижинского"         → stem="нижинск"
m4: "Альфонсо XIII"      → special (монарх)
m5: "Григорьев"          → stem="григорьев"
m6: "Чарли Чаплином"     → stem="чаплин", first="Чарли"
m7: "Чаплин"             → stem="чаплин"
m8: "М.Ф. Ларионова"     → stem="ларионов", inits=["М", "Ф"]
m9: "А.К. Томилиной"     → stem="томилин", inits=["А", "К"]
m10: "Э. Пёрвиенс"       → stem="пёрвиенс", init="Э"
m11: "Э. Кэмпбелла"      → stem="кэмпбелл", init="Э"
m12: "Ш. Мино"           → stem="мино", init="Ш"
m13: "Л. Лопуховой"      → stem="лопухов", init="Л"
```

**Сгенерированные кандидаты:**
```
(m2, m3) → "Вацлав Нижинский" + "Нижинского" — одна фамилия, confidence=0.95
(m6, m7) → "Чарли Чаплином" + "Чаплин" — одна фамилия, confidence=0.95
```

**Остальные mentions не имеют кандидатов на объединение** — каждый станет отдельной группой.

---

## Этап 2: Упрощённый промпт для LLM

### 2.1 Новая задача LLM

Вместо "сгруппируй все mentions" → "подтверди/отклони предложенные объединения"

### 2.2 Упрощённый промпт

```python
SYSTEM_PROMPT_GROUPING_V2 = """
Ты проверяешь, относятся ли пары упоминаний к одному человеку.

На входе:
- Список mentions (упоминаний людей)
- Список кандидатов на объединение (пары mentions)

На выходе:
- Для каждого кандидата: подтверждение (true) или отклонение (false)

ПРАВИЛА:
1. Подтверждай объединение ТОЛЬКО если уверен, что это один человек
2. Отклоняй, если есть сомнения или противоречия в контексте
3. Проверяй контекст: "Нижинский женился" и "выступления Нижинского" — один человек

Примеры:
- "Вацлав Нижинский" + "Нижинского" в контексте одного балета → true
- "И.И. Иванов" + "П.П. Иванов" → false (разные инициалы)
- "Григорьев" упомянут дважды как режиссёр → true (один контекст)
"""
```

### 2.3 Формат входа для LLM

```json
{
  "mentions": [
    {"id": "m2", "text": "Вацлав Нижинский", "context": "первый танцовщик Вацлав Нижинский женился..."},
    {"id": "m3", "text": "Нижинского", "context": "участие в спектаклях Нижинского"},
    {"id": "m6", "text": "Чарли Чаплином", "context": "встреча с Чарли Чаплином состоялась..."},
    {"id": "m7", "text": "Чаплин", "context": "Чаплин (в полицейской форме)..."}
  ],
  "candidates": [
    {"a": "m2", "b": "m3", "reason": "Совпадение фамилии: Нижинск*"},
    {"a": "m6", "b": "m7", "reason": "Совпадение фамилии: Чаплин*"}
  ]
}
```

### 2.4 Формат выхода от LLM

```json
{
  "decisions": [
    {"a": "m2", "b": "m3", "same_person": true, "confidence": 0.95},
    {"a": "m6", "b": "m7", "same_person": true, "confidence": 0.95}
  ]
}
```

---

## Этап 3: Построение групп из решений

### 3.1 Union-Find для объединения

```python
def build_groups_from_decisions(
    mentions: list[PersonMention],
    decisions: list[MergeDecision],
) -> list[MentionGroup]:
    """
    Использует Union-Find для построения групп.

    1. Каждый mention — отдельная компонента
    2. Для каждого подтверждённого решения — union
    3. Результат: связные компоненты = группы
    """
    dsu = DSU(mention_ids)

    for decision in decisions:
        if decision.same_person:
            dsu.union(decision.a, decision.b)

    # Группируем mentions по компонентам
    components = defaultdict(list)
    for mid in mention_ids:
        root = dsu.find(mid)
        components[root].append(mid)

    return [
        MentionGroup(
            group_id=f"g{i}",
            mention_ids=members,
            canonical_name=select_canonical(members, mentions_map),
            is_person=True,
            confidence=compute_group_confidence(members, decisions),
        )
        for i, members in enumerate(components.values())
    ]
```

### 3.2 Выбор canonical_name

```python
def select_canonical(
    member_ids: list[str],
    mentions_map: dict[str, PersonMention],
) -> str:
    """
    Выбирает наиболее полную форму имени из группы.

    Приоритет:
    1. Полное ФИО (Иван Иванович Иванов)
    2. Имя + Фамилия (Иван Иванов)
    3. Инициалы + Фамилия (И.И. Иванов)
    4. Только фамилия (Иванов)

    Приводит к именительному падежу.
    """
```

---

## Этап 4: Полный пайплайн

```
mentions (13 штук)
    │
    ▼
┌─────────────────────────────┐
│ 1. Парсинг mentions         │
│    extract_name_components  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. Генерация кандидатов     │
│    generate_merge_candidates│
│    → [(m2,m3), (m6,m7)]     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. LLM валидация            │  ← Упрощённый промпт
│    validate_candidates      │
│    → [true, true]           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. Union-Find группировка   │
│    build_groups             │
└─────────────┬───────────────┘
              │
              ▼
groups (11 штук)
    g1: [m1] "Дягилев"
    g2: [m2, m3] "Вацлав Нижинский"
    g3: [m4] "Альфонсо XIII"
    g4: [m5] "Григорьев"
    g5: [m6, m7] "Чарли Чаплин"
    g6: [m8] "М.Ф. Ларионов"
    g7: [m9] "А.К. Томилина"
    g8: [m10] "Э. Пёрвиенс"
    g9: [m11] "Э. Кэмпбелл"
    g10: [m12] "Ш. Мино"
    g11: [m13] "Л. Лопухова"
```

---

## Этап 5: Реализация

### 5.1 Новые файлы/функции

```
src/tools/
├── mention_grouper.py          # Текущий (оставить для fallback)
├── name_parser.py              # НОВЫЙ: парсинг имён
│   ├── parse_mention()
│   ├── extract_last_name_stem()
│   └── extract_initials()
├── candidate_generator.py      # НОВЫЙ: генерация кандидатов
│   ├── generate_merge_candidates()
│   └── MergeCandidate
└── grouper_v2.py               # НОВЫЙ: новый grouper
    ├── validate_candidates_async()  # LLM валидация
    ├── build_groups_from_decisions()
    └── group_mentions_v2_async()    # Основная функция
```

### 5.2 Модели данных

```python
# src/models.py (дополнения)

class ParsedMention(BaseModel):
    mention_id: str
    text_span: str
    last_name_stem: str | None
    first_name: str | None
    first_initial: str | None
    patronymic: str | None
    patronymic_initial: str | None
    has_title: bool = False  # "патриарх", "князь"
    title: str | None = None

class MergeCandidate(BaseModel):
    mention_id_a: str
    mention_id_b: str
    algorithmic_confidence: float
    reason: str

class MergeDecision(BaseModel):
    mention_id_a: str
    mention_id_b: str
    same_person: bool
    confidence: float

class CandidateValidationResponse(BaseModel):
    note_id: str
    decisions: list[MergeDecision]
```

### 5.3 Порядок реализации

1. **name_parser.py** — парсинг имён (без LLM, чистый алгоритм)
2. **candidate_generator.py** — генерация кандидатов (без LLM)
3. **Тесты** для парсера и генератора
4. **grouper_v2.py** — новый grouper с LLM валидацией
5. **Интеграция** в people_extractor.py
6. **Тесты** end-to-end

---

## Преимущества нового подхода

| Аспект | Старый подход | Новый подход |
|--------|---------------|--------------|
| Детерминированность | Низкая | Высокая (кандидаты алгоритмические) |
| Потеря mentions | Часто | Невозможно (fallback на отдельные группы) |
| Ложные объединения | Часто | Редко (LLM только подтверждает) |
| Нагрузка на LLM | Высокая | Низкая (только пары) |
| Стабильность | Низкая | Высокая |
| Отладка | Сложная | Простая (видны кандидаты и решения) |

---

## Fallback стратегия

Если LLM недоступен или возвращает ошибку:
1. Все кандидаты с `algorithmic_confidence >= 0.9` автоматически подтверждаются
2. Остальные mentions становятся отдельными группами
3. Логируем warning о fallback режиме

```python
def fallback_decisions(candidates: list[MergeCandidate]) -> list[MergeDecision]:
    return [
        MergeDecision(
            mention_id_a=c.mention_id_a,
            mention_id_b=c.mention_id_b,
            same_person=c.algorithmic_confidence >= 0.9,
            confidence=c.algorithmic_confidence,
        )
        for c in candidates
    ]
```
