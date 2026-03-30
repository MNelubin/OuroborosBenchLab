"""
bench_cli.py — Headless benchmark entry point for Ouroboros.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

class TranscriptLogger:
    """Writes tool calls and LLM messages to a JSONL file in PinchBench format."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._total_cost = 0.0
        self._total_input = 0
        self._total_output = 0

    def log(self, entry: dict):
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_user(self, content: str):
        # PinchBench format: type=message, message={role, content}
        self.log({"type": "message", "message": {"role": "user", "content": content}})

    def log_assistant(self, content: str, usage: dict | None = None):
        msg_obj = {"role": "assistant", "content": content}
        entry = {"type": "message", "message": msg_obj}
        if usage:
            # Usage goes top-level or inside message? PinchBench usually top-level or meta
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

async def run_bench_task(prompt, workspace, model, timeout, transcript_out):
    logger = TranscriptLogger(transcript_out)
    start = time.time()
    exit_code = 0

    try:
        os.environ["OUROBOROS_MODEL"] = model
        os.environ["OUROBOROS_BENCH_MODE"] = "1"
        
        from ouroboros.agent import Agent
        agent = Agent()
        
        # Log user prompt first
        logger.log_user(prompt)

        # Run agent
        try:
            result = await asyncio.wait_for(
                agent.run_bench(task=prompt, logger=logger, workspace=workspace),
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
        logger.log({"type": "error", "message": str(e)})
        logger.log_done("crash")
        exit_code = 1
    finally:
        logger.close()

    return exit_code

def main():
    p = argparse.ArgumentParser(description="Ouroboros headless benchmark runner")
    p.add_argument("--prompt",         required=True)
    p.add_argument("--workspace",      required=True)
    p.add_argument("--model",          required=True)
    p.add_argument("--timeout",        type=int, default=120)
    p.add_argument("--transcript-out", required=True)
    args = p.parse_args()

    code = asyncio.run(run_bench_task(
        prompt         = args.prompt,
        workspace      = args.workspace,
        model          = args.model,
        timeout        = args.timeout,
        transcript_out = args.transcript_out,
    ))
    sys.exit(code)

if __name__ == "__main__":
    main()
