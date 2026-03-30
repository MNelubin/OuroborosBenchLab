import argparse, json, logging, os, re, sys, tempfile, time, yaml
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))
from lib_agent_ouroboros import ensure_agent_exists, prepare_task_workspace, execute_ouroboros_task, cleanup_agent_sessions
from lib_tasks import Task
from lib_grading import grade_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def load_tasks(tasks_dir, suite="all", task_ids=None):
    tasks = []
    for f in sorted(Path(tasks_dir).glob("task_*.md")):
        content = f.read_text()
        fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not fm: continue
        try: meta = yaml.safe_load(fm.group(1))
        except: continue
        body = content[fm.end():]
        
        def get_sec(title):
            m = re.search(rf"##\s+{re.escape(title)}\s*\n(.*?)(?=\n##\s|\Z)", body, re.DOTALL)
            return m.group(1).strip() if m else ""

        # workspace_files are in YAML frontmatter, not a body section
        workspace_files = meta.get("workspace_files") or []

        # lib_grading.py extracts ```python ... ``` from the raw section text
        grade_body = get_sec("Automated Checks") or get_sec("Grading Function")
        judge_rubric = get_sec("Judge Rubric") or get_sec("Rubric")
        expected_behavior = get_sec("Expected Behavior")

        prompt = get_sec("Prompt")

        t = Task(
            id=meta.get("id", f.stem),
            name=meta.get("name", f.stem),
            category=meta.get("category", "general"),
            grading_type=meta.get("grading_type", "automated"),
            timeout=int(meta.get("timeout_seconds", meta.get("timeout", 120))),
            prompt=prompt,
            workspace_files=workspace_files,
            automated_checks=grade_body,
            judge_rubric=judge_rubric,
            llm_judge_rubric=judge_rubric,
            expected_behavior=expected_behavior,
            grading_weights=meta.get("grading_weights") or {},
        )
        
        if t.is_openclaw_specific: continue
        if suite == "automated-only" and t.grading_type not in ("automated", "hybrid"): continue
        if task_ids and t.id not in task_ids: continue
        tasks.append(t)
    return tasks

def run_task(task, agent_id, model, ws_root, verbose=False, timeout_multiplier=1.0,
             judge_model=None, output_dir=None, proxy_url=None):
    ws = Path(ws_root) / task.id
    prepare_task_workspace(task, str(ws))

    if verbose:
        log.info(f"--- PROMPT START ---\n{task.prompt}\n--- PROMPT END ---")

    effective_timeout = max(int(task.timeout * timeout_multiplier), task.timeout)
    ex = execute_ouroboros_task(
        agent_id=agent_id, prompt=task.prompt, timeout=effective_timeout,
        model_name=model, workspace_path=str(ws), proxy_url=proxy_url
    )

    if verbose and ex.transcript:
        log.info("--- TRANSCRIPT START ---")
        for entry in ex.transcript:
            t = entry.get("type")
            if t == "message":
                role = entry.get("message", {}).get("role", "?")
                content = entry.get("message", {}).get("content", "")
                log.info(f"[{role.upper()}]:\n{content}\n")
            elif t == "tool_use":
                log.info(f"[TOOL {entry.get('name')}]:\n{json.dumps(entry.get('input', {}), ensure_ascii=False, indent=2)}\n")
            elif t == "tool_result":
                log.info(f"[RESULT {entry.get('name')}]:\n{entry.get('output', '')}\n")
            elif t == "error":
                log.info(f"[ERROR]: {entry.get('message')}")
        log.info("--- TRANSCRIPT END ---")

    if ex.status == "success":
        try:
            grade_kwargs = {"task": task, "execution_result": {**vars(ex), "workspace": str(ws)}, "skill_dir": ws}
            if judge_model:
                grade_kwargs["fast_judge_model"] = judge_model
            gr = grade_task(**grade_kwargs)
            score = gr.score / max(gr.max_score, 1)
            breakdown, notes = gr.breakdown, gr.notes
            if verbose:
                log.info(f"GRADER RESULT: score={score:.2f} breakdown={breakdown} notes={notes}")
        except Exception as e:
            log.error(f"Grading failed for {task.id}: {e}")
            import traceback
            traceback.print_exc()
            score, breakdown, notes = 0.0, {}, f"grading error: {e}"
    else:
        score, breakdown, notes = 0.0, {}, f"failed: {ex.status}"
        if ex.stderr:
            log.error(f"Container stderr: {ex.stderr[:500]}")

    result = {
        "task_id": task.id, "task_name": task.name,
        "score": round(score, 4), "status": ex.status,
        "duration": ex.duration, "cost_usd": ex.usage.get("cost", 0),
        "breakdown": breakdown, "notes": notes, "stderr": ex.stderr,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Full transcript JSONL
        tr_path = out / f"{task.id}.jsonl"
        with tr_path.open("w", encoding="utf-8") as f:
            for entry in ex.transcript:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Task result JSON
        (out / f"{task.id}.result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"  Saved: {tr_path}")

    cleanup_agent_sessions(agent_id)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tasks-dir", default="tasks")
    p.add_argument("--suite", default="automated-only")
    p.add_argument("--task-ids", nargs="+")
    p.add_argument("--verbose", action="store_true", help="Show prompts and transcripts")
    p.add_argument("--timeout-multiplier", type=float, default=1.0,
                    help="Scale all task timeouts (e.g. 2.0 for slow free-tier models)")
    p.add_argument("--judge-model", default=None,
                    help="Fast judge: direct OpenRouter API call with this model "
                         "(e.g. anthropic/claude-haiku-4-5-20251001). "
                         "Without this flag, original PinchBench judge path is used (claude-opus via Docker).")
    p.add_argument("--output-dir", default=None,
                    help="Directory to save full transcripts and results")
    p.add_argument("--proxy-url", default=None,
                    help="SOCKS/HTTP proxy URL to pass to the agent (e.g. socks5://user:pass@ip:port)")
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"): sys.exit("ERROR: OPENROUTER_API_KEY not set")
    if not os.environ.get("OPENAI_API_KEY"): sys.exit("ERROR: OPENAI_API_KEY not set (required for web_search tool)")

    tasks = load_tasks(args.tasks_dir, args.suite, args.task_ids)
    if not tasks: sys.exit("No tasks loaded")

    agent_id = ensure_agent_exists(args.model, None)
    ws_root = tempfile.mkdtemp(prefix="ouro_bench_")

    all_results = []

    log.info(f"Starting benchmark for {len(tasks)} tasks...")
    for i, task in enumerate(tasks, 1):
        log.info(f"[{i}/{len(tasks)}] {task.id}")
        res = run_task(task, agent_id, args.model, ws_root,
                       verbose=args.verbose, timeout_multiplier=args.timeout_multiplier,
                       judge_model=args.judge_model, output_dir=args.output_dir, proxy_url=args.proxy_url)
        all_results.append(res)
        log.info(f"  Score: {res['score']:.2f}  Status: {res['status']}")
        if res['status'] != 'success':
            log.error(f"  STDERR: {res['stderr'][:500]}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Saved aggregated results to {out / 'results.json'}")

if __name__ == "__main__":
    main()
