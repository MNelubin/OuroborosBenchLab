"""
bench_cli.py — Headless benchmark entry point for Ouroboros.

Runs the FULL Ouroboros agent (SYSTEM.md + BIBLE.md + identity + tools)
in headless mode, identical to how it runs via Colab/Telegram.
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time


class TranscriptLogger:
    """Writes tool calls and LLM messages to a JSONL file in PinchBench format."""

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._total_cost = 0.0
        self._total_input = 0
        self._total_output = 0

    def log(self, entry: dict):
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_user(self, content: str):
        self.log({"type": "message", "message": {"role": "user", "content": content}})

    def log_assistant(self, content: str, usage: dict | None = None):
        msg_obj = {"role": "assistant", "content": content}
        entry = {"type": "message", "message": msg_obj}
        if usage:
            entry["usage"] = usage
            self._total_cost += usage.get("cost", 0.0)
            self._total_input += usage.get("input", 0)
            self._total_output += usage.get("output", 0)
        self.log(entry)

    def log_tool_use(self, name: str, input_data: dict):
        self.log({"type": "tool_use", "name": name, "input": input_data})

    def log_tool_result(self, name: str, output: str):
        self.log({"type": "tool_result", "name": name, "output": str(output)[:2000]})

    def log_done(self, reason: str = "task_complete"):
        self.log({
            "type": "done", "reason": reason,
            "total_cost": round(self._total_cost, 6),
        })

    def close(self):
        self._file.close()


async def run_bench_task(prompt, workspace, model, timeout, transcript_out,
                         repo_dir, drive_root):
    logger = TranscriptLogger(transcript_out)
    start = time.time()
    exit_code = 0

    try:
        # --- Set env vars exactly as colab_launcher does ---
        os.environ["OUROBOROS_MODEL"] = model
        os.environ["OUROBOROS_MODEL_CODE"] = model
        os.environ["OUROBOROS_BENCH_MODE"] = "1"
        os.environ["OUROBOROS_EVOLUTION_ENABLED"] = "0"
        os.environ["OUROBOROS_CONSCIOUSNESS_ENABLED"] = "0"

        # --- Bootstrap drive structure before agent init ---
        # (OuroborosAgent constructor runs startup verification which reads state.json)
        drive_root_path = pathlib.Path(drive_root)
        for sub in ("state", "logs", "memory", "index", "locks", "archive"):
            (drive_root_path / sub).mkdir(parents=True, exist_ok=True)
        state_path = drive_root_path / "state" / "state.json"
        if not state_path.exists():
            state_path.write_text(json.dumps({
                "spent_usd": 0, "total_budget": 0,
                "current_branch": "bench", "current_sha": "bench",
            }), encoding="utf-8")

        # --- Create Agent with proper Env (mirrors Colab layout) ---
        from ouroboros.agent import OuroborosAgent, Env

        env = Env(
            repo_dir=pathlib.Path(repo_dir),
            drive_root=drive_root_path,
            branch_dev="bench",
        )
        agent = OuroborosAgent(env=env)

        # --- Augment prompt with workspace context ---
        # Agent tools default to repo_dir (/app); workspace files are at /workspace.
        # Shell tool accepts absolute paths, so we tell the agent where to look.
        # /no_think disables chain-of-thought on Qwen3 thinking models so they use tools
        # instead of just describing what they would do.
        augmented_prompt = (
            f"[Bench workspace: {workspace}]\n"
            f"IMPORTANT: Complete the task by calling tools. Do NOT just describe what you would do.\n"
            f"Files are in {workspace}/. Use run_shell with absolute paths:\n"
            f'  run_shell(["cat", "{workspace}/filename.txt"])  # read\n'
            f'  run_shell(["bash", "-c", "cat > {workspace}/output.txt <<\'EOF\'\\ncontent\\nEOF"])  # write\n'
            f"\n"
            f"{prompt}"
        )

        # Log user prompt first (original, not augmented — for grading clarity)
        logger.log_user(prompt)

        # Run agent with full context (SYSTEM.md + BIBLE.md + identity + tools)
        try:
            result = await asyncio.wait_for(
                agent.run_bench(task=augmented_prompt, logger=logger, workspace=workspace),
                timeout=timeout,
            )
            logger.log_done("task_complete")
        except asyncio.TimeoutError:
            logger.log_done("timeout")
            exit_code = 2

    except ImportError as e:
        logger.log({"type": "error", "message": f"ImportError: {e}"})
        logger.log_done("import_error")
        exit_code = 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.log({"type": "error", "message": str(e)})
        logger.log_done("crash")
        exit_code = 1
    finally:
        logger.close()

    elapsed = round(time.time() - start, 2)
    print(f"[bench_cli] Finished in {elapsed}s, exit_code={exit_code}", file=sys.stderr)
    return exit_code


def main():
    p = argparse.ArgumentParser(description="Ouroboros headless benchmark runner")
    p.add_argument("--prompt",         required=True,
                    help="Task prompt to send to the agent")
    p.add_argument("--workspace",      required=True,
                    help="Path to task workspace directory")
    p.add_argument("--model",          required=True,
                    help="Model ID (e.g. anthropic/claude-sonnet-4.6)")
    p.add_argument("--timeout",        type=int, default=120,
                    help="Timeout in seconds")
    p.add_argument("--transcript-out", required=True,
                    help="Path to write JSONL transcript")
    p.add_argument("--repo-dir",       default="/app",
                    help="Path to ouroboros repo (contains BIBLE.md, prompts/, etc.)")
    p.add_argument("--drive-root",     default="/drive",
                    help="Path to drive-like root (state/, logs/, memory/)")
    args = p.parse_args()

    code = asyncio.run(run_bench_task(
        prompt         = args.prompt,
        workspace      = args.workspace,
        model          = args.model,
        timeout        = args.timeout,
        transcript_out = args.transcript_out,
        repo_dir       = args.repo_dir,
        drive_root     = args.drive_root,
    ))
    sys.exit(code)


if __name__ == "__main__":
    main()
