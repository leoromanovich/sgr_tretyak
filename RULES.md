1. Структура проекта
```
project_root/
  data/
    pages/                 # исходные md-страницы
    obsidian/
      images/              # уже есть
      persons/             # сюда пишем людей
      items/               # сюда пишем произведения
  cache/
    notes.jsonl            # кэш извлечённых метаданных по страницам
    persons_local.jsonl    # кандидаты-персоны по страницам
    persons_clusters.json  # кластеризация глобальных персон
  src/
    config.py
    llm_client.py
    models.py              # Pydantic-схемы для SGR
    tools/
      note_metadata.py
      people_extractor.py
      name_normalizer.py
      person_matcher.py
      item_extractor.py
      note_linker.py
      person_note_generator.py
      item_note_generator.py
    orchestrator.py        # главный пайплайн / CLI
```

2. Libraries

```
pip install \
  openai \
  pydantic \
  pydantic-settings \
  python-frontmatter \
  markdown-it-py \
  rich \
  typer \
  orjson
```

Пояснения:
•	openai — клиент к твоему vLLM-серверу (OpenAI-совместимый).
•	pydantic — строгие модели для SGR (валидация structured output).
•	pydantic-settings — конфиг (BASE_URL, MODEL, пути и т.п.).
•	python-frontmatter — удобный парсинг YAML frontmatter + body из md.
•	markdown-it-py — если понадобится более тонкий парсинг структуры страниц.
•	rich — нормальный логгер/прогресс.
•	typer — простой CLI-обвес для запуска этапов пайплайна.
•	orjson — быстрый JSON, удобно для кэша.


3. LLM-клиент и конфиг

3.1. Конфиг (config.yaml)
```
llm:
  api_key: "dummy"                      # твой ключ для локального прокси
  base_url: "http://localhost:8000/v1"   # твой openai-совместимый сервер
  model: "models/Qwen/Qwen3-14B-FP8"
  max_tokens: 32768
  temperature: 0.2

blocking:
  year_bucket_size: 10                  # размер корзины по годам (десятилетие)
  max_block_size: 200                   # максимум кандидатов в блоке
  fallback_key: "last_name_first_name"  # более строгий ключ при слишком большом блоке
  keys:
    last_name_first_initial: true
    last_name_patronymic_initial: true
    last_name_year_bucket: true
    last_name_first_name: true
```

пути:
•	PAGES_DIR = data/pages
•	OBS_PERSONS_DIR = data/obsidian/persons
•	OBS_ITEMS_DIR = data/obsidian/items


3.2. Клиент (llm_client.py)
Создаёшь thin-wrapper над OpenAI:
•	инициализация с base_url и api_key,
•	helper-функция call_chat(messages, response_format=None, tools=None, tool_choice="auto"…),
•	helper для SGR + Pydantic: call_and_parse(model_class, messages, response_format_schema) — валидирует JSON через Pydantic и выбрасывает исключение при несоответствии.

4. Pydantic-модели (схемы для SGR)
В models.py описываешь строго:
4.1. Метаданные заметки
```
NoteMetadata:
  note_id: str
  path: Path
  primary_year: int | None
  year_range: tuple[int,int] | None
  location: str | None
  topic: str | None
  reliability: float
```


4.2. Локальные кандидаты-персоны (на уровне одной страницы)
```
PersonLocal:
    note_id: str
    local_person_id: str
    surface_forms: list[str]
    canonical_name_in_note: str
    is_person: bool
    confidence: float
    note_year_context: int | None
    note_year_source: Literal["inline", "note_metadata", "unknown"]
    snippet_evidence: list[SnippetEvidence]

SnippetEvidence:
    snippet: str
    reasoning: str
```
После нормализации:
```
PersonLocalNormalized(PersonLocal):
    normalized_full_name: str | None
    name_parts:
    last_name: str | None
    first_name: str | None
    patronymic: str | None
    abbreviation_links:
    list[AbbreviationLink]

AbbreviationLink:
    from_form: str
    to_form: str
    confidence: float
    reasoning: str
```
4.3. Глобальные персоны и матчинги
```
PersonCandidate:
    note_id: str
    local_person_id: str
    normalized_full_name: str | None
    name_parts: ...
    note_year_context: int | None
    snippets: list[str]
    person_confidence: float

PersonMatchDecision:
    left: (note_id, local_person_id)
    right: (note_id, local_person_id)
    relation: Literal["same_person", "different_person", "unknown"]
    confidence: float
    reasoning: str
```
После кластеризации:
```
GlobalPerson:
    global_person_id: str
    canonical_full_name: str
    year_key: str | None        # "1812", "1812–1815"
    episodes: list[EpisodeRef]
    disambiguation_key: str | None  # "генерал", "чиновник" и т.п.
```
Аналогично можно завести ItemCandidate, GlobalItem для произведений.

5. Реализация «тулов» (LLM-вызовы)
Каждый тул — это просто отдельный промпт/функция + Pydantic-схема.

5.1. note_metadata_extractor
•	Вход: исходный markdown страницы.
•	System-промпт:
•	объяснить, что заметка описывает эпизод/период;
•	извлечь primary_year / year_range, location, topic;
•	нельзя придумывать года.
•	Выход: NoteMetadata (через response_format + json_schema).

Используется в первом проходе по всем файлам data/pages.

⸻

5.2. people_extractor
•	Вход: markdown + NoteMetadata.
•	System-промпт:
•	найти все упоминания людей;
•	объединить формы, которые очевидно относятся к одному человеку;
•	отфильтровать организации / топонимы (is_person = false).
•	Выход: список PersonLocal.

Сохраняешь в cache/persons_local.jsonl.

⸻

5.3. name_normalizer

Работает по одной заметке.
•	Вход: список PersonLocal для note_id + сам текст заметки (для дополнительного контекста).
•	Задача:
•	собрать normalized_full_name, если полное ФИО где-то есть;
•	разложить на name_parts;
•	связать И.И. Иванов ↔ Иван Иванович Иванов в рамках заметки.
•	Выход: список PersonLocalNormalized.

Храним поверх PersonLocal (тот же jsonl, но с дополнительными полями или новый файл).

⸻

5.4. person_matcher (между заметками)
•	Вход: два PersonCandidate (из разных или тех же заметок).
•	Принцип:
•	используется для кандидатов с «подозрительно похожими» ФИО (одинаковая фамилия и первые буквы имени/отчества) и пересекающимися или близкими годами.
•	предварительный отбор пар — чисто программно:
•	block по фамилии;
•	по первым буквам имени/инициалам;
•	по year_context (разница в годах < N, скажем 30).
•	LLM-промпт:
•	описываешь вход как два кратких профиля + их snippet’ы;
•	требуешь строгое решение same_person | different_person | unknown с объяснением и confidence.
•	Выход: PersonMatchDecision.

Дальше кластера строишь через DSU/union-find поверх same_person с confidence >= threshold.

⸻

5.5. person_note_generator

На уровне кластера.
•	Вход: GlobalPerson-заготовка (id, canonical_full_name, список эпизодов с выдержками текста).
•	LLM-промпт:
•	сформировать содержимое Obsidian-заметки:
•	frontmatter:
•	full_name
•	episodes (список ссылок на note_id и краткое описание)
•	year_key, disambiguation_key, теги (people, history/...)
•	тело:
•	раздел «Роль в контексте»
•	раздел «Эпизоды»
•	раздел «Связанные лица» (список [[Другие персоны]], если популярны в тех же эпизодах).
•	Выход: markdown-строка.
•	Файл: пишешь в data/obsidian/persons/{generated_filename}.md.

⸻

5.6. item_extractor и item_note_generator

Аналогично для «произведений»:
•	extractor:
•	из страницы выделяет произведения (картины, книги, события как «items»),
•	привязывает к годам и людям.
•	note_generator:
•	формирует заметки в data/obsidian/items.
•	учитывает images/ (можно в frontmatter прописывать image: ../images/filename.jpg).

⸻

5.7. note_linker

Перелинковка исходных pages:
•	Вход: исходный markdown, маппинг:
•	mention_span -> [[PersonFileName|текст]]
•	аналогично для items: [[ItemFileName|текст]].
•	Вариант 1 (детерминированный):
•	не звать LLM, а заменять через свою логику, если у тебя есть точные surface_forms и их позиции.
•	Вариант 2 (через LLM + SGR):
•	даёшь список canonical_name + соответствующий файл.
•	просишь модель переписать заметку, оборачивая только известные упоминания, ничего не придумывая.
•	схема на выходе: { linked_note_markdown: str }.
•	Пишешь поверх старого файла или в отдельную папку data/pages_linked/.

⸻

6. Оркестратор (агент)

orchestrator.py — CLI с несколькими subcommands (через Typer):
1.	scan-notes
•	проход по data/pages,
•	для каждой страницы:
•	note_metadata_extractor;
•	people_extractor;
•	name_normalizer;
•	(опционально) item_extractor;
•	сохраняешь результаты в cache/*.jsonl.
2.	cluster-persons
•	читаешь всех PersonLocalNormalized с is_person=True, confidence>=0.8;
•	строишь набор PersonCandidate;
•	генерируешь кандидаты-пары через блокинг (ключи по фамилии+инициалам/году и т.п., см. config.yaml);
•	для каждой пары запускаешь person_matcher;
•	строишь кластеры (union-find);
•	создаёшь GlobalPerson записи и сохраняешь в cache/persons_clusters.json.
3.	generate-person-notes
•	по GlobalPerson вызываешь person_note_generator;
•	пишешь файлы в data/obsidian/persons.
4.	generate-item-notes
•	аналогично для items.
5.	link-pages
•	строишь маппинг:
•	из note_id + local_person_id → GlobalPerson → имя файла;
•	для каждой страницы запускаешь note_linker;
•	сохраняешь изменённые markdown-файлы.
6.	(опционально) full-run — оркестрация цепочкой: scan → cluster → generate-* → link.

⸻

7. Практические замечания
    1.	Идемпотентность.
          Всегда пиши промежуточные результаты в cache/ и не перезапускай всё подряд, если уже есть валидный jsonl. Для LLM-дорогих шагов (person_matcher) это критично.
    2.	Пороги и фильтрация.
          По умолчанию:
          •	PersonLocal.confidence < 0.7 — можно отбрасывать;
          •	PersonMatchDecision.confidence < 0.75 → unknown, не мержим.
          Чуть позже подстроишь под свои данные.
    3.	Без ручной валидации.
          Если совсем не хочешь руками править:
          •	добавь финальный шаг sanity-check (ещё один LLM-вызов) для проверки:
          •	нет ли ссылок в страницах на несуществующие .md и наоборот;
          •	нет ли кластеров, где в reasoning явно конфликтуют годы (модель может сама заметить).
    4.	Логи.
          Используй rich для логов и прогресса (особенно на cluster-persons), чтобы понимать, где и что движется.

⸻

Если хочешь, следующим шагом можем расписать конкретные Pydantic-модели и пример одного-двух промптов для SGR (note_metadata_extractor + people_extractor), чтобы у тебя была жёсткая стартовая точка.
