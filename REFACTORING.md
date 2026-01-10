# План тестирования пайплайна извлечения и линковки персон

## Общая структура

```
tests/
├── conftest.py                    # Общие фикстуры
├── fixtures/
│   ├── notes/                     # Исходные заметки для тестов
│   │   ├── 8331.md
│   │   ├── 8332.md
│   │   └── ...
│   ├── expected/                  # Ожидаемые результаты
│   │   ├── mentions/              # Ожидаемые упоминания
│   │   ├── persons/               # Ожидаемые персоны
│   │   ├── normalized/            # Ожидаемые нормализованные имена
│   │   └── linked/                # Ожидаемые линки
│   └── golden/                    # "Золотые" файлы из кэша (исправленные вручную)
├── test_mention_extraction.py
├── test_mention_grouping.py
├── test_name_normalization.py
├── test_person_candidates.py
├── test_clustering.py
├── test_note_linking.py
└── test_e2e.py                    # End-to-end тесты
```

---

## Этап 1: Извлечение упоминаний (mention_extractor.py)

### 1.1 Для разработчика тестов

**Файл:** `tests/test_mention_extraction.py`

**Что тестируем:**
- Все упоминания людей найдены (recall)
- Нет галлюцинаций (precision)
- `text_span` точно соответствует тексту в заметке
- `context_snippet` содержит контекст

**Структура теста:**
```python
import pytest
from src.tools.mention_extractor import extract_mentions
from tests.fixtures.loader import load_note, load_expected_mentions

class TestMentionExtraction:
    
    @pytest.mark.parametrize("note_id", ["8331", "8332", "8346"])
    def test_all_mentions_found(self, note_id):
        """Проверяет, что все ожидаемые упоминания найдены."""
        note_text = load_note(note_id)
        expected = load_expected_mentions(note_id)
        
        result = extract_mentions(note_id, note_text, metadata)
        
        found_spans = {m.text_span for m in result.mentions}
        expected_spans = {e["text_span"] for e in expected["must_find"]}
        
        missing = expected_spans - found_spans
        assert not missing, f"Не найдены упоминания: {missing}"
    
    def test_no_hallucinations(self, note_id):
        """Проверяет, что все найденные упоминания есть в тексте."""
        note_text = load_note(note_id)
        result = extract_mentions(note_id, note_text, metadata)
        
        for mention in result.mentions:
            assert mention.text_span in note_text, \
                f"Галлюцинация: '{mention.text_span}' нет в тексте"
    
    def test_known_failures(self):
        """Регрессионные тесты на известные ошибки."""
        # Загружаем из fixtures/regressions/mentions.yaml
        ...
```

**Вспомогательные функции (conftest.py):**
```python
def text_contains_form(text: str, form: str, fuzzy: bool = True) -> bool:
    """Проверяет наличие формы в тексте (с учётом морфологии)."""
    ...
```

### 1.2 Для подготовки данных

**Формат файла:** `tests/fixtures/expected/mentions/8331.yaml`

```yaml
note_id: "8331"
source_file: "8331.md"

# Обязательные упоминания (должны быть найдены)
must_find:
  - text_span: "Павла Корина"
    likely_person: true
  - text_span: "Павел Корин"
    likely_person: true
  - text_span: "П.Д. Корин"
    likely_person: true
  - text_span: "Павел Дмитриевич"
    likely_person: true
  - text_span: "Корин"
    likely_person: true
    comment: "одиночная фамилия в контексте"
  - text_span: "патриарха Тихона"
    likely_person: true
  - text_span: "Корина"
    likely_person: true
    comment: "родительный падеж"

# Не должны быть найдены (организации, места, абстракции)
must_not_find:
  - text_span: "Успенского собора"
    reason: "это место, не человек"
  - text_span: "духовенства"
    reason: "абстрактная группа"

# Известные проблемы (TODO — пока пропускаем)
known_issues:
  - text_span: "Е.В. Вучетича"
    issue: "упоминание в метаинформации, не в основном тексте"
```

**Как создать из кэша:**
```bash
# 1. Запустить пайплайн на заметке
python -m src.cli note-people 8331.md > /tmp/mentions_8331.json

# 2. Просмотреть и отредактировать
# 3. Сохранить как fixtures/expected/mentions/8331.yaml
```

---

## Этап 2: Группировка упоминаний (mention_grouper.py)

### 2.1 Для разработчика тестов

**Файл:** `tests/test_mention_grouping.py`

**Что тестируем:**
- Упоминания одного человека объединены в одну группу
- Разные люди не смешаны
- `canonical_name` — самая полная форма
- `canonical_name` в именительном падеже (!)

**Структура теста:**
```python
class TestMentionGrouping:
    
    def test_same_person_grouped(self, note_id):
        """Проверяет, что варианты одного имени в одной группе."""
        expected = load_expected_groups(note_id)
        result = group_mentions(...)
        
        for expected_group in expected["groups"]:
            # Найти группу по canonical_name
            found = find_group_by_canonical(result.groups, expected_group["canonical"])
            assert found, f"Группа не найдена: {expected_group['canonical']}"
            
            # Проверить, что все формы в группе
            for form in expected_group["must_contain"]:
                assert form in found.mention_ids
    
    def test_canonical_nominative_case(self, note_id):
        """Проверяет, что canonical_name в именительном падеже."""
        result = group_mentions(...)
        
        for group in result.groups:
            # Простая проверка: не должно заканчиваться на типичные окончания косвенных падежей
            assert not group.canonical_name.endswith(('а', 'у', 'ом', 'ой', 'ых')), \
                f"Возможно не именительный падеж: {group.canonical_name}"
            # Или использовать pymorphy2 для точной проверки
    
    def test_different_persons_separated(self, note_id):
        """Проверяет, что разные люди в разных группах."""
        ...
```

### 2.2 Для подготовки данных

**Формат файла:** `tests/fixtures/expected/groups/8331.yaml`

```yaml
note_id: "8331"

groups:
  - canonical_name: "Павел Дмитриевич Корин"
    canonical_nominative: true  # Флаг для проверки падежа
    must_contain_forms:
      - "Павла Корина"
      - "Павел Корин"
      - "П.Д. Корин"
      - "Павел Дмитриевич"
      - "Корин"
      - "Корина"
    is_person: true
    
  - canonical_name: "патриарх Тихон"  # Именительный!
    canonical_nominative: true
    must_contain_forms:
      - "патриарха Тихона"
    is_person: true

# Проверка разделения
must_be_separate:
  - ["Павел Дмитриевич Корин", "патриарх Тихон"]
```

---

## Этап 3: Нормализация имён (name_normalizer.py)

### 3.1 Для разработчика тестов

**Файл:** `tests/test_name_normalization.py`

**Что тестируем:**
- `normalized_full_name` максимально полное
- `name_parts` корректно разобраны
- Все имена в именительном падеже
- `abbreviation_links` связывают сокращения с полными формами

**Структура теста:**
```python
class TestNameNormalization:
    
    @pytest.mark.parametrize("note_id,person_id,expected_name", [
        ("8331", "p1", "Павел Дмитриевич Корин"),
        ("8331", "p2", "Тихон"),  # патриарх — это роль, не часть имени
        ("8332", "p1", "Павел Дмитриевич Корин"),
        ("8332", "p2", "Алексей Викторович Щусев"),
    ])
    def test_normalized_full_name(self, note_id, person_id, expected_name):
        result = normalize_people_in_file(...)
        person = find_person(result, person_id)
        assert person.normalized_full_name == expected_name
    
    def test_name_parts_extracted(self):
        """Проверяет разбор на фамилию/имя/отчество."""
        result = normalize_people_in_file(...)
        person = find_person(result, "p1")
        
        assert person.name_parts.last_name == "Корин"
        assert person.name_parts.first_name == "Павел"
        assert person.name_parts.patronymic == "Дмитриевич"
    
    def test_nominative_case(self):
        """Все части имени в именительном падеже."""
        result = normalize_people_in_file(...)
        
        for person in result.people:
            if person.name_parts.last_name:
                # Фамилия не должна быть в родительном падеже
                assert not person.name_parts.last_name.endswith('а'), \
                    f"Фамилия в род. падеже?: {person.name_parts.last_name}"
```

### 3.2 Для подготовки данных

**Формат файла:** `tests/fixtures/expected/normalized/8331.yaml`

```yaml
note_id: "8331"

persons:
  - local_person_id: "p1"
    normalized_full_name: "Павел Дмитриевич Корин"
    name_parts:
      last_name: "Корин"
      first_name: "Павел"
      patronymic: "Дмитриевич"
    surface_forms:
      - "Павла Корина"
      - "Павел Корин"
      - "П.Д. Корин"
      - "Павел Дмитриевич"
      - "Корин"
      - "Корина"
    role: "художник"
    
  - local_person_id: "p2"
    normalized_full_name: "Тихон"
    name_parts:
      last_name: null
      first_name: "Тихон"
      patronymic: null
    surface_forms:
      - "патриарха Тихона"
    role: "патриарх"
```

**Как создать из кэша `persons_local_normalized.jsonl`:**
```bash
# Фильтруем записи по note_id
cat cache/persons_local_normalized.jsonl | \
  jq -c 'select(.note_id == "8331")' | \
  yq -P > tests/fixtures/expected/normalized/8331.yaml

# Затем вручную исправить ошибки (падежи, недостающие части)
```

---

## Этап 4: Формирование кандидатов (scan_people.py)

### 4.1 Для разработчика тестов

**Файл:** `tests/test_person_candidates.py`

**Что тестируем:**
- Все персоны из нормализации попали в кандидаты
- `confidence_bucket` корректно вычислен
- Поля для блокинга заполнены (`normalized_last_name`, `first_initial`, etc.)

```python
class TestPersonCandidates:
    
    def test_all_persons_become_candidates(self):
        """Каждая персона с is_person=true становится кандидатом."""
        ...
    
    def test_blocking_fields_populated(self):
        """Поля для блокинга заполнены корректно."""
        candidates = load_person_candidates()
        
        for c in candidates:
            if c.name_parts.last_name:
                assert c.normalized_last_name is not None
                # Нормализация: ё→е, нижний регистр
                assert c.normalized_last_name == c.normalized_last_name.lower()
                assert 'ё' not in c.normalized_last_name
```

### 4.2 Для подготовки данных

Можно использовать существующий `cache/persons_local_normalized.jsonl` напрямую, отфильтровав нужные записи.

---

## Этап 5: Кластеризация (cluster_people.py)

### 5.1 Для разработчика тестов

**Файл:** `tests/test_clustering.py`

**Что тестируем:**
- Один человек из разных заметок объединён в один кластер
- Разные люди с похожими именами не смешаны
- `canonical_full_name` кластера корректен

```python
class TestClustering:
    
    def test_same_person_clustered(self):
        """Корин из 8331 и 8332 — один кластер."""
        clusters = cluster_people(...)
        
        korin_cluster = find_cluster_by_name(clusters, "Павел Дмитриевич Корин")
        assert korin_cluster is not None
        
        member_notes = {m.split(':')[0] for m in korin_cluster.members}
        assert "8331" in member_notes
        assert "8332" in member_notes
    
    def test_different_persons_not_mixed(self):
        """Разные люди не в одном кластере."""
        clusters = cluster_people(...)
        
        # Найти кластеры
        korin = find_cluster_by_name(clusters, "Корин")
        tihon = find_cluster_by_name(clusters, "Тихон")
        
        # Проверить, что они разные
        assert korin.global_person_id != tihon.global_person_id
```

### 5.2 Для подготовки данных

**Формат файла:** `tests/fixtures/expected/clusters.yaml`

```yaml
clusters:
  - canonical_full_name: "Павел Дмитриевич Корин"
    must_contain_candidates:
      - "8331:p1"
      - "8332:p1"
      - "8346:p1"  # если есть
    
  - canonical_full_name: "Тихон"
    must_contain_candidates:
      - "8331:p2"
    must_not_contain:
      - "8332:p1"  # Это Корин, не Тихон

# Пары, которые НЕ должны быть в одном кластере
must_be_separate:
  - ["Павел Дмитриевич Корин", "Тихон"]
  - ["Павел Дмитриевич Корин", "Алексей Викторович Щусев"]
```

**Как создать из кэша `persons_clusters.json`:**
```bash
cat cache/persons_clusters.json | \
  jq '.persons[] | {canonical_full_name, members}' | \
  yq -P > tests/fixtures/expected/clusters.yaml
  
# Затем вручную проверить и исправить
```

---

## Этап 6: Линковка в заметках (note_linker.py)

### 6.1 Для разработчика тестов

**Файл:** `tests/test_note_linking.py`

**Что тестируем:**
- Все формы имён залинкованы
- Линки ведут на правильные файлы персон
- Нет двойных линков `[[[[...]]]`
- Не линкуются alt-тексты изображений

```python
import re

class TestNoteLinking:
    
    def test_all_forms_linked(self, note_id):
        """Все surface_forms персоны залинкованы."""
        expected = load_expected_links(note_id)
        
        source_text = load_note(note_id)
        linked_text = apply_links(source_text, ...)
        
        for link_spec in expected["must_link"]:
            form = link_spec["form"]
            target = link_spec["target"]
            
            # Ищем [[target|form]] или [[target|<вариация формы>]]
            pattern = rf'\[\[{re.escape(target)}\|[^\]]*\]\]'
            assert re.search(pattern, linked_text), \
                f"Форма '{form}' не залинкована на '{target}'"
    
    def test_no_double_brackets(self, note_id):
        """Нет вложенных скобок [[[[...]]]]."""
        linked_text = apply_links(...)
        assert '[[[[' not in linked_text
        assert ']]]]' not in linked_text
    
    def test_image_alt_not_linked(self, note_id):
        """Alt-текст изображений не линкуется."""
        linked_text = apply_links(...)
        
        # Паттерн: ![alt](path) — внутри alt не должно быть [[]]
        for match in re.finditer(r'!\[([^\]]*)\]', linked_text):
            alt_text = match.group(1)
            assert '[[' not in alt_text, \
                f"Alt-текст содержит линк: {alt_text}"
    
    def test_author_in_meta_linked(self, note_id):
        """Автор в метаинформации залинкован."""
        linked_text = apply_links(...)
        
        # Ищем строку "**Автор:**" и проверяем, что после неё есть линк
        author_line = re.search(r'\*\*Автор:\*\*\s*(.+)', linked_text)
        if author_line:
            assert '[[' in author_line.group(1), \
                "Автор в метаинформации не залинкован"
```

### 6.2 Для подготовки данных

**Формат файла:** `tests/fixtures/expected/linked/8331.yaml`

```yaml
note_id: "8331"

# Формы, которые ДОЛЖНЫ быть залинкованы
must_link:
  - form: "Корин Павел"
    target: "Павел_Дмитриевич_Корин"
    location: "meta"  # в метаинформации
    
  - form: "Павла Корина"
    target: "Павел_Дмитриевич_Корин"
    location: "body"
    
  - form: "Павел Корин"
    target: "Павел_Дмитриевич_Корин"
    location: "body"
    
  - form: "П.Д. Корин"
    target: "Павел_Дмитриевич_Корин"
    
  - form: "патриарха Тихона"
    target: "Тихон"
    
  - form: "Е.В. Вучетича"
    target: "Евгений_Викторович_Вучетич"
    location: "meta"

# Формы, которые НЕ должны линковаться
must_not_link:
  - form: "Успенского собора"
    reason: "это место"
  - form: "духовенства"
    reason: "абстрактная группа"

# Проверка итогового текста (опционально)
expected_fragments:
  - "**Автор:** [[Павел_Дмитриевич_Корин|Корин Павел]]"
  - "[[Павел_Дмитриевич_Корин|Павла Корина]]"
  - "[[Тихон|патриарха Тихона]]"
```

**Как создать из обработанных заметок:**
```bash
# 1. Взять выходной файл из data/obsidian/items/
# 2. Извлечь все линки:
grep -oE '\[\[[^\]]+\]\]' data/obsidian/items/Реквием_8331.md

# 3. Сравнить с исходным текстом и найти пропущенные
diff <(grep -oE '[А-Я][а-я]+' 8331.md | sort -u) \
     <(grep -oE '\[\[[^\]]+\|([^\]]+)\]\]' linked.md | sort -u)
```

---

## Этап 7: End-to-End тесты

### 7.1 Для разработчика тестов

**Файл:** `tests/test_e2e.py`

```python
class TestEndToEnd:
    
    def test_full_pipeline_single_note(self, tmp_path):
        """Полный пайплайн для одной заметки."""
        # Копируем тестовую заметку
        shutil.copy("tests/fixtures/notes/8331.md", tmp_path / "pages")
        
        # Запускаем пайплайн
        scan_people_over_pages(...)
        clusters = cluster_people(...)
        write_person_notes(clusters)
        link_persons_in_pages(...)
        
        # Проверяем результат
        linked_file = tmp_path / "obsidian/items/Реквием_8331.md"
        assert linked_file.exists()
        
        content = linked_file.read_text()
        assert "[[Павел_Дмитриевич_Корин|" in content
    
    def test_regression_known_bugs(self):
        """Регрессия на известные баги."""
        # Загрузить из fixtures/regressions/
        ...
```

---

## Вспомогательные утилиты

### Загрузчик фикстур (`tests/fixtures/loader.py`)

```python
import yaml
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

def load_note(note_id: str) -> str:
    path = FIXTURES_DIR / "notes" / f"{note_id}.md"
    return path.read_text(encoding="utf-8")

def load_expected_mentions(note_id: str) -> dict:
    path = FIXTURES_DIR / "expected/mentions" / f"{note_id}.yaml"
    return yaml.safe_load(path.read_text())

def load_expected_groups(note_id: str) -> dict:
    path = FIXTURES_DIR / "expected/groups" / f"{note_id}.yaml"
    return yaml.safe_load(path.read_text())

# ... аналогично для других типов
```

### Скрипт генерации фикстур из кэша

```bash
#!/bin/bash
# scripts/generate_fixtures.sh

NOTE_ID=$1

# Из persons_local_normalized.jsonl
cat cache/persons_local_normalized.jsonl | \
  jq -c "select(.note_id == \"$NOTE_ID\")" | \
  python -c "
import sys, json, yaml
records = [json.loads(line) for line in sys.stdin]
print(yaml.dump({'note_id': '$NOTE_ID', 'persons': records}, allow_unicode=True))
" > tests/fixtures/expected/normalized/${NOTE_ID}.yaml

echo "Создан tests/fixtures/expected/normalized/${NOTE_ID}.yaml"
echo "Проверьте и исправьте вручную!"
```

---

## Чеклист для добавления нового тестового кейса

### Когда найдена ошибка:

1. **Определить этап**, на котором возникла ошибка
2. **Создать/дополнить фикстуру** для этого этапа
3. **Добавить в `known_issues`** если пока не исправляем
4. **Или добавить в `must_find`/`must_link`** если исправляем

### Пример: "Корин Павел" не линкуется

```yaml
# tests/fixtures/expected/linked/8331.yaml
must_link:
  - form: "Корин Павел"
    target: "Павел_Дмитриевич_Корин"
    location: "meta"
    bug_id: "ISSUE-001"  # ссылка на issue
```

---

## Приоритеты реализации

| Приоритет | Этап | Причина |
|-----------|------|---------|
| 1 | Линковка | Видимый результат, легко проверить |
| 2 | Нормализация | Критично для корректных имён |
| 3 | Группировка | Влияет на падежи |
| 4 | Извлечение упоминаний | Базовый recall |
| 5 | Кластеризация | Сложнее тестировать |
| 6 | E2E | После базовых тестов |

---

## Быстрый старт

```bash
# 1. Создать структуру
mkdir -p tests/fixtures/{notes,expected/{mentions,groups,normalized,linked},golden}

# 2. Скопировать тестовые заметки
cp data/pages/8331.md tests/fixtures/notes/
cp data/pages/8332.md tests/fixtures/notes/

# 3. Сгенерировать базовые фикстуры из кэша
./scripts/generate_fixtures.sh 8331
./scripts/generate_fixtures.sh 8332

# 4. Вручную исправить фикстуры (падежи, пропущенные формы)

# 5. Запустить тесты
pytest tests/ -v
```
