from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Task:
    id: str
    name: str
    category: str
    grading_type: str
    timeout: int
    prompt: str
    workspace_files: List[Dict[str, Any]] = field(default_factory=list)
    automated_checks: str = ""
    judge_rubric: str = ""
    llm_judge_rubric: str = ""
    expected_behavior: str = ""
    grading_weights: Dict[str, float] = field(default_factory=dict)
    grading_criteria: List[str] = field(default_factory=list)
    difficulty: str = "medium"

    @property
    def task_id(self):
        """Alias for compatibility with lib_grading.py"""
        return self.id

    @property
    def grade_fn(self):
        """Alias for PinchBench compatibility (maps to automated_checks)"""
        return self.automated_checks

    @property
    def is_openclaw_specific(self):
        # Tasks skipped for Ouroboros:
        # task_21 — content is about OpenClaw ecosystem (skill registry PDF)
        # task_14 — prompt requires /install humanizer (OpenClaw slash command)
        # task_08 — grader checks transcript for readFile/toolCall (OpenClaw tool names); run_shell never matches
        # task_10 — same: grader checks toolCall+readfile for read_config; always 0 for Ouroboros
        return self.id in {
            "task_21_openclaw_comprehension",
            "task_14_humanizer",
            "task_08_memory",
            "task_10_workflow",
        }
