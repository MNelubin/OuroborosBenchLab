# OuroborosBenchLab

Стенд для оценки качества агента **Ouroboros** на задачах бенчмарка **PinchBench**.

Позволяет запускать одни и те же задачи на Ouroboros и на оригинальном агенте OpenClaw, сравнивать результаты и отслеживать прогресс модели.

---

## Концепция

**Ouroboros** — самомодифицирующийся AI-агент (работает через Telegram/Colab), который управляет собственным кодом, памятью и стратегией. В боевом режиме он эволюционирует сам по себе.

**PinchBench** — набор из 24 задач (task_00..task_22), разработанных командой OpenClaw для оценки агентов. Задачи делятся на:
- `automated` — проверяются скриптами (файлы, вычисления, форматы)
- `llm_judge` — оценивает LLM-судья по рубрике
- `hybrid` — и то, и другое

**OuroborosBenchLab** соединяет их: запускает Ouroboros в изолированном Docker-контейнере, подаёт ему задачи из PinchBench и оценивает ответы той же логикой грейдинга.

```
ouroboros/        ← исходный код агента + Dockerfile.bench
pinchbench/       ← оригинальный PinchBench (задачи, грейдинг, OpenClaw)
runner/           ← наш адаптер: запуск Ouroboros на задачах PinchBench
```

---

## Требования

- Docker
- Python 3.10+
- `OPENROUTER_API_KEY` (для вызовов моделей через OpenRouter)

---

## 1. Настройка окружения

```bash
cd runner
cp .env.example .env
# Вписать OPENROUTER_API_KEY в .env
```

Или просто экспортировать:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

---

## 2. Сборка Docker-образа

```bash
cd ouroboros
docker build -f Dockerfile.bench -t ouroboros-bench:latest .
```

Образ содержит весь код агента (BIBLE.md, SYSTEM.md, prompts/, ouroboros/), точку входа `bench_cli.py` и структуру директорий `/workspace`, `/drive`, `/transcripts`.

---

## 3. Запуск бенчмарка

```bash
cd runner
python benchmark_ouroboros.py \
    --model anthropic/claude-sonnet-4-6 \
    --suite automated-only \
    --output-dir results/my_run
```

### Основные флаги

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--model` | обязателен | Модель в формате OpenRouter (`anthropic/claude-sonnet-4-6`) |
| `--suite` | `automated-only` | `automated-only` / `all` |
| `--task-ids` | все | Запустить только конкретные задачи: `--task-ids task_00 task_08` |
| `--timeout-multiplier` | `1.0` | Множитель таймаутов (для медленных free-tier моделей) |
| `--judge-model` | — | Быстрый LLM-судья напрямую через OpenRouter (напр. `anthropic/claude-haiku-4-5-20251001`). Без флага используется оригинальный Docker-путь PinchBench с Claude Opus |
| `--output-dir` | — | Сохранить транскрипты и результаты в директорию |
| `--proxy-url` | — | SOCKS/HTTP прокси для агента (напр. `socks5://user:pass@ip:port`) |
| `--verbose` | — | Выводить полные промпты и транскрипты |

### Просмотр результатов

После запуска с `--output-dir`:

```
results/my_run/
  task_00_sanity.jsonl        ← полный транскрипт (вся переписка агента)
  task_00_sanity.result.json  ← оценка и breakdown по чекам
  task_08_memory.jsonl
  task_08_memory.result.json
  results.json                ← агрегированные результаты всех задач
```

---

## 4. Сравнение с OpenClaw (PinchBench)

Скрипт `run_both.sh` запускает оба агента на одной модели и выводит сравнение:

```bash
cd runner
bash run_both.sh anthropic/claude-sonnet-4-6 automated-only
```

Или вручную через `compare_results.py`:

```bash
python compare_results.py pinchbench/results/oc_result.json results/my_run/results.json
```

---

## 5. Задачи

Задачи лежат в `runner/tasks/` (копия из `pinchbench/tasks/` с адаптациями).
Каждый файл — Markdown с YAML-фронтматтером:

```yaml
---
id: task_00_sanity
name: Sanity Check
category: basic
grading_type: automated
timeout_seconds: 60
---
## Prompt
...
## Automated Checks
```python
...
```
```

Шаблон новой задачи: [runner/tasks/TASK_TEMPLATE.md](runner/tasks/TASK_TEMPLATE.md)

---

## Отфильтрованные задачи (OpenClaw-specific)

Следующие задачи **пропускаются автоматически** (помечены как `is_openclaw_specific` в `lib_tasks.py`) — они либо требуют инфраструктуру OpenClaw, либо их грейдинг завязан на инструменты OpenClaw и всегда даёт 0 для Ouroboros:

| Задача | Причина пропуска |
|--------|-----------------|
| `task_21_openclaw_comprehension` | Содержание задачи — вопросы по PDF-отчёту об экосистеме OpenClaw (статистика реестра навыков, SKILL.md, WebSocket API). Бессмысленно для любого не-OpenClaw агента. |
| `task_14_humanizer` | Промпт требует команды `/install humanizer` — это OpenClaw-specific slash command. Задача тестирует установку навыков из реестра OpenClaw. |
| `task_08_memory` | Грейдер проверяет транскрипт на наличие вызова инструмента `readFile` (OpenClaw). Ouroboros читает файлы через `run_shell`, поэтому критерий `read_notes` всегда 0. |
| `task_10_workflow` | Аналогично: грейдер ищет `readFile`/`read_file` toolCall в транскрипте для проверки чтения `config.json`. Для Ouroboros критерий `read_config` всегда 0. |

> **Примечание:** `task_11_clawdhub` несмотря на название — обычная задача создания Python-проекта, никак не связанная с OpenClaw. Она включена в бенчмарк.

---

## 6. Удалённый запуск

Стенд развёрнут на сервере. Подключение:

```bash
ssh root@100.64.0.30   # → затем ssh root@192.168.10.119
cd /opt/ouroborosbench/runner
```

---

## Структура репозитория

```
OuroborosBenchLab/
├── ouroboros/              ← агент Ouroboros
│   ├── Dockerfile.bench    ← образ для headless запуска
│   ├── ouroboros/          ← Python-пакет агента
│   │   ├── agent.py        ← OuroborosAgent, run_bench()
│   │   ├── bench_cli.py    ← CLI-точка входа бенчмарка
│   │   └── tools/          ← инструменты агента
│   ├── prompts/SYSTEM.md   ← системный промпт
│   └── BIBLE.md            ← конституция агента
│
├── pinchbench/             ← оригинальный PinchBench (не трогаем)
│   ├── tasks/              ← 24 задачи
│   └── scripts/            ← runner и grader OpenClaw
│
└── runner/                 ← наш адаптер
    ├── benchmark_ouroboros.py  ← основной скрипт запуска
    ├── lib_agent_ouroboros.py  ← Docker-адаптер
    ├── lib_grading.py          ← грейдинг (PinchBench-совместимый)
    ├── lib_tasks.py            ← модель задачи
    ├── tasks/                  ← задачи (адаптация из pinchbench/tasks/)
    ├── run_both.sh             ← запуск обоих бенчей + сравнение
    ├── compare_results.py      ← сравнение двух results.json
    └── .env.example            ← шаблон переменных окружения
```
