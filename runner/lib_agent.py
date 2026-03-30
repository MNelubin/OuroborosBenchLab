"""
lib_agent.py — Shim (adapter) for PinchBench compatibility.
Redirects calls from lib_grading.py to lib_agent_ouroboros.py.
"""

# Импортируем реализации из Ouroboros-адаптера
from lib_agent_ouroboros import (
    ensure_agent_exists,
    run_ouroboros_prompt,
    execute_ouroboros_task,
    prepare_task_workspace,
    cleanup_agent_sessions
)

# Функция slugify, которую ожидает lib_grading
def slugify_model(model_name: str) -> str:
    return model_name.replace("/", "-").replace(":", "-").replace(".", "-")

# Алиас для совместимости: lib_grading вызывает run_openclaw_prompt,
# но мы перенаправляем это на Ouroboros-реализацию.
run_openclaw_prompt = run_ouroboros_prompt
