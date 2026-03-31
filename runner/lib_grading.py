"""
PinchBench grading engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib_agent import ensure_agent_exists, run_openclaw_prompt, slugify_model
from lib_tasks import Task


logger = logging.getLogger(__name__)


DEFAULT_JUDGE_MODEL = "openrouter/anthropic/claude-opus-4.5"
DEFAULT_JUDGE_AGENT_PREFIX = "bench-judge"
DEFAULT_JUDGE_TIMEOUT_SECONDS = 180


@dataclass
class GradeResult:
    task_id: str
    score: float
    max_score: float
    grading_type: str
    breakdown: Dict[str, float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "max_score": self.max_score,
            "grading_type": self.grading_type,
            "breakdown": self.breakdown,
            "notes": self.notes,
        }


def grade_task(
    *,
    task: Task,
    execution_result: Dict[str, Any],
    skill_dir: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_agent_prefix: str = DEFAULT_JUDGE_AGENT_PREFIX,
    judge_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
    fast_judge_model: Optional[str] = None,
    verbose: bool = False,
) -> GradeResult:
    grading_type = task.grading_type
    if verbose:
        logger.info("   [VERBOSE] Grading task %s with type: %s", task.task_id, grading_type)
        logger.info("   [VERBOSE] Execution status: %s", execution_result.get("status", "unknown"))
    
    if grading_type == "automated":
        result = _grade_automated(task, execution_result, verbose=verbose)
        if verbose:
            logger.info("   [VERBOSE] Automated grade breakdown: %s", result.breakdown)
        return result
    if grading_type == "llm_judge":
        result = _grade_llm_judge(
            task=task,
            execution_result=execution_result,
            judge_model=judge_model,
            judge_agent_prefix=judge_agent_prefix,
            judge_timeout_seconds=judge_timeout_seconds,
            skill_dir=skill_dir,
            fast_judge_model=fast_judge_model,
            verbose=verbose,
        )
        if verbose:
            logger.info("   [VERBOSE] LLM judge breakdown: %s", result.breakdown)
        return result
    if grading_type == "hybrid":
        auto_result = _grade_automated(task, execution_result, verbose=verbose)
        llm_result = _grade_llm_judge(
            task=task,
            execution_result=execution_result,
            judge_model=judge_model,
            judge_agent_prefix=judge_agent_prefix,
            judge_timeout_seconds=judge_timeout_seconds,
            skill_dir=skill_dir,
            fast_judge_model=fast_judge_model,
            verbose=verbose,
        )
        return _combine_grades(task, auto_result, llm_result)
    raise ValueError(f"Unknown grading type: {grading_type}")


def _normalize_transcript_for_grading(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Ouroboros JSONL transcript to OpenClaw-compatible format so that
    grading functions (written for OpenClaw) don't crash on Ouroboros output.

    Ouroboros:  content is a plain string; tool_use/tool_result are top-level events.
    OpenClaw:   content is a list of typed items; tool calls are embedded in assistant message.
    """
    normalized = []
    for event in transcript:
        t = event.get("type")
        if t == "message":
            msg = event.get("message", event)
            content = msg.get("content", "")
            if isinstance(content, str):
                # Wrap string content as OpenClaw text item
                norm_content: List[Any] = [{"type": "text", "text": content}]
            else:
                norm_content = content
            normalized.append({
                "type": "message",
                "message": {**msg, "content": norm_content},
            })
        elif t == "tool_use":
            # Represent as an assistant message with a toolCall content item
            normalized.append({
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "toolCall",
                        "name": event.get("name", ""),
                        "params": event.get("input", {}),
                    }],
                },
            })
        elif t == "tool_result":
            # Represent as a toolResult role message
            normalized.append({
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": str(event.get("output", ""))}],
                },
            })
        else:
            normalized.append(event)
    return normalized


def _grade_automated(task: Task, execution_result: Dict[str, Any], verbose: bool = False) -> GradeResult:
    grading_code = _extract_grading_code(task)
    if not grading_code:
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="No automated grading code found",
        )

    namespace: Dict[str, Any] = {}
    exec(grading_code, namespace)
    grade_func = namespace.get("grade")
    if not callable(grade_func):
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="Automated grading function missing",
        )

    scores = grade_func(
        _normalize_transcript_for_grading(execution_result.get("transcript", [])),
        execution_result.get("workspace", ""),
    )
    if not isinstance(scores, dict):
        scores = {}
    
    if verbose:
        logger.info("   [VERBOSE] Automated grading scores: %s", scores)

    total = _average_scores(scores)
    return GradeResult(
        task_id=task.task_id,
        score=total,
        max_score=1.0,
        grading_type="automated",
        breakdown=_normalize_score_dict(scores),
        notes="",
    )


def _call_judge_llm(prompt: str, model: str, timeout: float = 180) -> str:
    """Direct OpenRouter API call for judge — no Docker container needed."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — judge returns empty response")
        return ""

    # Strip "openrouter/" prefix if present (OpenRouter API expects just the model ID)
    clean_model = model.removeprefix("openrouter/")

    payload = json.dumps({
        "model": clean_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    proxy_url = os.environ.get("OUROBOROS_PROXY_URL", "")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            import httpx
            client_kwargs = {"timeout": int(timeout)}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
            with httpx.Client(**client_kwargs) as client:
                resp = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    content=payload,
                    headers=headers,
                )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0))
                wait = retry_after if retry_after > 0 else min(2 ** attempt * 5, 60)
                logger.warning("Judge rate-limited (429), retry %d/%d in %ds — model=%s",
                               attempt + 1, max_retries, wait, clean_model)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.error("Judge LLM HTTP %s for model=%s: %s",
                             resp.status_code, clean_model, resp.text[:500])
                return ""
            data = resp.json()
            actual_model = data.get("model", "unknown")
            if actual_model != clean_model:
                logger.warning("   [JUDGE] OpenRouter routed to a DIFFERENT model: requested=%s actual=%s",
                               clean_model, actual_model)
            else:
                logger.info("   [JUDGE] OpenRouter confirmed model: %s", actual_model)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Judge LLM call failed (model=%s): %s", clean_model, e)
            return ""
    logger.error("Judge LLM gave up after %d retries (model=%s)", max_retries, clean_model)
    return ""


def _grade_llm_judge(
    *,
    task: Task,
    execution_result: Dict[str, Any],
    judge_model: str,
    judge_agent_prefix: str,
    judge_timeout_seconds: float,
    skill_dir: Path,
    fast_judge_model: Optional[str] = None,
    verbose: bool = False,
) -> GradeResult:
    transcript_summary = _summarize_transcript(execution_result.get("transcript", []))
    if verbose:
        logger.info("   [VERBOSE] Transcript summary for judge (first 1000 chars):\n%s", transcript_summary[:1000])
    rubric = task.llm_judge_rubric or _format_grading_criteria(task)
    prompt = _build_judge_prompt(task, transcript_summary, rubric)

    if fast_judge_model:
        # Fast path: direct OpenRouter API call (no Docker container)
        logger.info("   [JUDGE] Fast mode — calling %s directly via OpenRouter", fast_judge_model)
        raw_text = _call_judge_llm(prompt, fast_judge_model, judge_timeout_seconds)
        logger.info("   [JUDGE] Raw response (%d chars): %s", len(raw_text), raw_text[:800])
        raw_parsed = _parse_judge_text(raw_text)
    else:
        # Original PinchBench path: run judge via Ouroboros Docker (claude-opus by default)
        agent_id = _ensure_judge_agent(judge_agent_prefix, judge_model, skill_dir)
        judge_workspace = Path(f"/tmp/pinchbench/judge/{task.task_id}")
        judge_result = run_openclaw_prompt(
            agent_id=agent_id,
            prompt=prompt,
            model_name=judge_model,
            workspace=judge_workspace,
            timeout_seconds=judge_timeout_seconds,
        )
        if isinstance(judge_result, str):
            logger.info("   [JUDGE] Raw response (%d chars): %s", len(judge_result), judge_result[:800])
            raw_parsed = _parse_judge_text(judge_result)
        else:
            raw_parsed = _parse_judge_response(judge_result.get("transcript", []))

    if not raw_parsed:
        logger.warning("   [JUDGE] Parse returned empty — score will be 0. Check raw response above.")

    parsed = _normalize_judge_response(raw_parsed)
    breakdown = parsed.get("scores", {})
    total = parsed.get("total")
    notes = parsed.get("notes", "")
    return GradeResult(
        task_id=task.task_id,
        score=float(total) if total is not None else 0.0,
        max_score=1.0,
        grading_type="llm_judge",
        breakdown=_normalize_score_dict(breakdown),
        notes=str(notes) if notes is not None else "",
    )


def _combine_grades(task: Task, auto_result: GradeResult, llm_result: GradeResult) -> GradeResult:
    weights = task.grading_weights or {"automated": 0.5, "llm_judge": 0.5}
    auto_weight = float(weights.get("automated", 0.5))
    llm_weight = float(weights.get("llm_judge", 0.5))
    total_weight = auto_weight + llm_weight
    if total_weight <= 0:
        auto_weight = llm_weight = 0.5
        total_weight = 1.0
    combined_score = (
        auto_result.score * auto_weight + llm_result.score * llm_weight
    ) / total_weight
    breakdown = {
        **{f"automated.{k}": v for k, v in auto_result.breakdown.items()},
        **{f"llm_judge.{k}": v for k, v in llm_result.breakdown.items()},
    }
    notes = " | ".join(filter(None, [auto_result.notes, llm_result.notes]))
    return GradeResult(
        task_id=task.task_id,
        score=combined_score,
        max_score=1.0,
        grading_type="hybrid",
        breakdown=breakdown,
        notes=notes,
    )


def _extract_grading_code(task: Task) -> str:
    if not task.automated_checks:
        return ""
    match = re.search(r"```python\s*(.*?)\s*```", task.automated_checks, re.DOTALL)
    if not match:
        return ""
    return match.group(1)


def _average_scores(scores: Dict[str, Any]) -> float:
    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_score_dict(scores: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in scores.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _format_grading_criteria(task: Task) -> str:
    if not task.grading_criteria:
        return ""
    return "\n".join(f"- {criterion}" for criterion in task.grading_criteria)


def _summarize_transcript(transcript: List[Dict[str, Any]]) -> str:
    """Summarize Ouroboros JSONL transcript (type: message/tool_use/tool_result/done)."""
    summary_parts: List[str] = []
    for event in transcript:
        t = event.get("type")
        if t == "message":
            msg = event.get("message", event)
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content)
            if role == "user":
                summary_parts.append(f"User: {content[:300]}")
            elif role == "assistant":
                summary_parts.append(f"Assistant: {content[:500]}")
        elif t == "tool_use":
            name = event.get("name", "unknown")
            inp = json.dumps(event.get("input", {}))[:150]
            summary_parts.append(f"Tool call: {name}({inp})")
        elif t == "tool_result":
            name = event.get("name", "unknown")
            out = str(event.get("output", ""))[:200]
            summary_parts.append(f"Tool result [{name}]: {out}")
    return "\n".join(summary_parts)


def _build_judge_prompt(task: Task, transcript_summary: str, rubric: str) -> str:
    return (
        "You are a grading function. Your ONLY job is to output a single JSON object.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT use any tools (no Read, Write, exec, or any other tool calls)\n"
        "- Do NOT create files or run commands\n"
        "- Do NOT write any prose, explanation, or commentary outside the JSON\n"
        "- Respond with ONLY a JSON object — nothing else\n\n"
        "Be a strict evaluator. Reserve 1.0 for genuinely excellent performance. "
        "An average acceptable completion should score around 0.6-0.7. "
        "Deduct points for unnecessary steps, verbose output, and inefficient tool usage.\n\n"
        "## Task\n"
        f"{task.prompt}\n\n"
        "## Expected Behavior\n"
        f"{task.expected_behavior}\n\n"
        "## Agent Transcript (summarized)\n"
        f"{transcript_summary}\n\n"
        "## Grading Rubric\n"
        f"{rubric}\n\n"
        "Score each criterion from 0.0 to 1.0.\n"
        'The "total" field must also be between 0.0 and 1.0, and it must be the arithmetic mean of the criterion scores, not their sum.\n\n'
        "Respond with ONLY this JSON structure (no markdown, no code fences, no extra text):\n"
        '{"scores": {"criterion_name": 0.0}, "total": 0.0, "notes": "brief justification"}'
    )


def _ensure_judge_agent(judge_agent_prefix: str, judge_model: str, skill_dir: Path) -> str:
    model_slug = slugify_model(judge_model)
    agent_id = f"{judge_agent_prefix}-{model_slug}"
    ensure_agent_exists(judge_model)  # verifies Docker image exists; returns agent_id
    return agent_id


def _parse_judge_response(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract assistant text from a transcript and parse JSON from it."""
    content_chunks: List[str] = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", event)
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            content_chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_chunks.append(item.get("text", ""))
                elif isinstance(item, str):
                    content_chunks.append(item)
    return _parse_judge_text("\n".join(content_chunks))


def _parse_judge_text(raw_text: str) -> Dict[str, Any]:
    """Parse a raw judge LLM response string into a structured dict."""
    raw_text = raw_text.strip()
    if not raw_text:
        return {}

    # First, try to extract JSON from code blocks (```json ... ```)
    code_block_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Find all potential JSON objects by looking for balanced braces
    # We'll extract chunks that start with { and try to parse them
    json_candidates: List[str] = []
    brace_depth = 0
    current_json = []
    for char in raw_text:
        if char == "{":
            if brace_depth == 0:
                current_json = []
            brace_depth += 1

        if brace_depth > 0:
            current_json.append(char)

        if char == "}":
            brace_depth -= 1
            if brace_depth == 0 and current_json:
                json_candidates.append("".join(current_json))

    # Try parsing from the last JSON object backwards (most recent response)
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "scores" in parsed:
                # Prefer JSON that has the expected structure
                return parsed
        except json.JSONDecodeError:
            continue

    # Try any valid JSON dict
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try to extract numeric scores from prose responses.
    # Models sometimes return "Total: 0.72" or "Overall score: 0.65" instead of JSON.
    score_pattern = re.search(
        r"(?:total|overall|final)\s*(?:score)?[:\s]*(0\.\d+|1\.0+)",
        raw_text,
        re.IGNORECASE,
    )
    if score_pattern:
        try:
            total = float(score_pattern.group(1))
            if 0.0 <= total <= 1.0:
                logger.warning(
                    "Fell back to regex score extraction from prose (total=%.2f)", total
                )
                return {"scores": {}, "total": total, "notes": "Score extracted from prose (JSON parse failed)"}
        except ValueError:
            pass

    logger.warning("Failed to parse judge JSON response")
    return {}


def _normalize_judge_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize judge response to expected format with 'scores', 'total', and 'notes'.
    
    Handles various response formats:
    - {"scores": {...}, "total": 0.9, "notes": "..."}  (expected)
    - {"criteria_scores": {...}, ...}  (Claude sometimes uses this)
    - {"score": 0.9, "justification": "..."}  (simplified format)
    """
    result: Dict[str, Any] = {"scores": {}, "total": None, "notes": ""}
    
    # Extract scores from various keys
    if "scores" in parsed:
        scores_data = parsed["scores"]
        if isinstance(scores_data, dict):
            # Handle nested structure: {"criterion": {"score": 0.9, "weight": 0.3}}
            for key, value in scores_data.items():
                if isinstance(value, dict) and "score" in value:
                    result["scores"][key] = float(value["score"]) if isinstance(value["score"], (int, float, str)) else value["score"]
                elif isinstance(value, (int, float)):
                    result["scores"][key] = value
    elif "criteria_scores" in parsed:
        # Handle Claude's alternate format
        criteria = parsed["criteria_scores"]
        if isinstance(criteria, dict):
            for key, value in criteria.items():
                if isinstance(value, dict) and "score" in value:
                    result["scores"][key] = value["score"]
                elif isinstance(value, (int, float)):
                    result["scores"][key] = value
    
    # Extract total score
    if "total" in parsed and parsed["total"] is not None:
        result["total"] = float(parsed["total"]) if isinstance(parsed["total"], (int, float)) else None
    elif "score" in parsed and isinstance(parsed["score"], (int, float)):
        result["total"] = float(parsed["score"])
    elif "overall_score" in parsed and isinstance(parsed["overall_score"], (int, float)):
        result["total"] = float(parsed["overall_score"])
    elif result["scores"]:
        # Calculate average if we have individual scores but no total
        values = [v for v in result["scores"].values() if isinstance(v, (int, float))]
        if values:
            result["total"] = sum(values) / len(values)

    # Some judge models return a summed total across criteria even though each
    # criterion is scored on a 0..1 scale. Normalize that back to a 0..1 mean.
    values = [v for v in result["scores"].values() if isinstance(v, (int, float))]
    if (
        values
        and result["total"] is not None
        and result["total"] > 1.0
        and all(0.0 <= float(v) <= 1.0 for v in values)
    ):
        result["total"] = sum(values) / len(values)
    
    # Extract notes/justification
    if "notes" in parsed:
        result["notes"] = str(parsed["notes"])
    elif "justification" in parsed:
        result["notes"] = str(parsed["justification"])
    elif "reasoning" in parsed:
        result["notes"] = str(parsed["reasoning"])
    
    return result
