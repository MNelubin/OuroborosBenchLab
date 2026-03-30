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
        return self.id in {"task_11_clawdhub", "task_21_openclaw_comprehension"}
