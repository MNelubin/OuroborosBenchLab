"""
lib_agent_ouroboros.py — Docker adapter for Ouroboros.
Mirror of PinchBench's lib_agent.py.
"""
import json, logging, os, shlex, subprocess, tempfile, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BENCH_IMAGE_NAME     = "ouroboros-bench"
BENCH_IMAGE_TAG      = "latest"
TRANSCRIPT_FILENAME  = "transcript.jsonl"
TRANSCRIPT_MAX_RETRIES = 6
TRANSCRIPT_RETRY_DELAY = 1.0


@dataclass
class ExecutionResult:
    status: str
    transcript: list
    usage: dict
    duration: float
    exit_code: int
    stderr: str = ""

    @property
    def success(self): return self.status == "success"


def ensure_agent_exists(model_name: str, workspace_path=None) -> str:
    tag = f"{BENCH_IMAGE_NAME}:{BENCH_IMAGE_TAG}"
    r = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Docker image {tag} not found.\n"
            "Run: bash /tmp/ouroborosbench/setup/fix_and_rebuild.sh"
        )
    agent_id = f"ouroboros-bench-{_slugify(model_name)}"
    log.debug(f"Agent ready: {agent_id}")
    return agent_id


def _slugify(s: str) -> str:
    return s.replace("/", "-").replace(":", "-").replace(".", "-")


def prepare_task_workspace(task, workspace_path: str):
    ws = Path(workspace_path)
    ws.mkdir(parents=True, exist_ok=True)
    # Look for binary assets in runner/assets/, fall back to pinchbench/assets/
    runner_dir = Path(__file__).parent
    assets_candidates = [
        runner_dir / "assets",
        runner_dir.parent / "pinchbench" / "assets",
    ]
    for wf in getattr(task, "workspace_files", []):
        if "path" in wf and "content" in wf:
            t = ws / wf["path"]
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(wf["content"], encoding="utf-8")
        elif "source" in wf and "dest" in wf:
            import shutil
            dest = ws / wf["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            for assets_dir in assets_candidates:
                src = assets_dir / wf["source"]
                if src.exists():
                    shutil.copy2(src, dest)
                    break
            else:
                log.warning(f"Asset not found: {wf['source']} (searched {assets_candidates})")


def execute_ouroboros_task(
    agent_id: str,
    prompt: str,
    timeout: int,
    model_name: Optional[str] = None,
    workspace_path: Optional[str] = None,
    proxy_url: Optional[str] = None,
) -> ExecutionResult:
    start = time.time()
    transcript_dir  = tempfile.mkdtemp(prefix="ouro_transcript_")
    transcript_path = Path(transcript_dir) / TRANSCRIPT_FILENAME

    ws = Path(workspace_path) if workspace_path else Path(tempfile.mkdtemp(prefix="ouro_ws_"))
    ws.mkdir(parents=True, exist_ok=True)

    if not model_name:
        model_name = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4-6")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    container_name = f"ouro-bench-{uuid.uuid4().hex[:8]}"

    env_vars = [
        "-e", f"OPENROUTER_API_KEY={api_key}",
        "-e", f"OUROBOROS_MODEL={model_name}",
        "-e", f"OUROBOROS_MODEL_CODE={model_name}",
        "-e", "OUROBOROS_BENCH_MODE=1",
        "-e", "OUROBOROS_EVOLUTION_ENABLED=0",
        "-e", "OUROBOROS_CONSCIOUSNESS_ENABLED=0",
    ]
    if openai_api_key:
        env_vars.extend(["-e", f"OPENAI_API_KEY={openai_api_key}"])
    
    # Check if proxy_url is passed or if it exists in the environment
    actual_proxy = proxy_url or os.environ.get("OUROBOROS_PROXY_URL")
    if actual_proxy:
        env_vars.extend(["-e", f"OUROBOROS_PROXY_URL={actual_proxy}"])

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{ws.resolve()}:/workspace",
        "-v", f"{transcript_dir}:/transcripts",
    ] + env_vars + [
        "--network", "bridge",
        f"{BENCH_IMAGE_NAME}:{BENCH_IMAGE_TAG}",
        "--prompt",         prompt,
        "--workspace",      "/workspace",
        "--model",          model_name,
        "--timeout",        str(timeout),
        "--transcript-out", f"/transcripts/{TRANSCRIPT_FILENAME}",
        "--repo-dir",       "/app",
        "--drive-root",     "/drive",
    ]

    log.info(f"Starting container: {container_name} (timeout={timeout}s)")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        exit_code = proc.returncode
        stderr    = proc.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired:
        exit_code = -1
        stderr    = "Container timeout"
        timed_out = True
        # kill immediately — stop may hang if container ignores SIGTERM
        try:
            subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=15)
        except Exception:
            pass

    duration   = round(time.time() - start, 2)
    transcript = _read_transcript(transcript_path)
    usage      = _extract_usage(transcript)

    if timed_out:
        status = "timeout"
    elif not transcript:
        status = "error"
    elif exit_code not in (0, -1):
        status = "error"
    else:
        status = "success"

    log.info(f"Done: status={status} duration={duration}s cost=${usage['cost']:.4f}")
    return ExecutionResult(
        status=status, transcript=transcript, usage=usage,
        duration=duration, exit_code=exit_code, stderr=stderr[:2000],
    )


def _read_transcript(path: Path, max_retries: int = TRANSCRIPT_MAX_RETRIES) -> list:
    for attempt in range(max_retries):
        if path.exists() and path.stat().st_size > 0:
            try:
                entries = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return entries
            except OSError:
                pass
        if attempt < max_retries - 1:
            time.sleep(TRANSCRIPT_RETRY_DELAY)
    return []


def _extract_usage(transcript: list) -> dict:
    u = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "request_count": 0}
    for e in transcript:
        if e.get("type") == "message" and e.get("role") == "assistant":
            raw = e.get("usage", {})
            u["input_tokens"]  += raw.get("input", 0)
            u["output_tokens"] += raw.get("output", 0)
            u["cost"]          += raw.get("cost", 0.0)
            u["request_count"] += 1
        elif e.get("type") == "done":
            tc = e.get("total_cost", 0.0)
            if tc > u["cost"]:
                u["cost"] = tc
    u["cost"] = round(u["cost"], 6)
    return u


def cleanup_agent_sessions(agent_id: str, workspace_path=None):
    try:
        r = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=ouro-bench-", "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=10,
        )
        ids = r.stdout.strip().split()
        if ids:
            subprocess.run(["docker", "rm", "-f"] + ids, capture_output=True, timeout=15)
    except Exception:
        pass


def run_ouroboros_prompt(agent_id, prompt, model_name=None, timeout=180,
                         workspace=None, timeout_seconds=None, **kwargs) -> str:
    # workspace is ignored (Ouroboros creates its own temp dir)
    # timeout_seconds is the PinchBench name for the same concept
    # model_name is optional — falls back to OUROBOROS_MODEL env var
    if model_name is None:
        model_name = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4-6")
    effective_timeout = timeout_seconds if timeout_seconds is not None else timeout
    with tempfile.TemporaryDirectory() as tmpdir:
        r = execute_ouroboros_task(
            agent_id=agent_id, prompt=prompt, timeout=effective_timeout,
            model_name=model_name, workspace_path=tmpdir,
        )
        for e in reversed(r.transcript):
            if e.get("type") == "message" and e.get("role") == "assistant":
                return e.get("content", "")
    return ""
