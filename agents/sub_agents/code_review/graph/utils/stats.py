"""
Workflow statistics for the orchestrator graph.

A single module-level `stats` singleton accumulates token usage and file-write
counters across a whole run, and prints the end-of-run summary. The orchestrator
node feeds it every LLM response via `record_tokens`; future nodes (coder,
validator) will feed it writes via `record_write`.

Ported verbatim from the standalone `orchestrator` project's `stats` module so
the planner logic in `orchestrator.py` keeps the same token-accounting hooks.
"""
import os
import time


class WorkflowStats:
    def __init__(self):
        self.start_time = 0.0
        self.lines_written = 0
        self.lines_deleted = 0
        self.files_created = 0
        self.files_overwritten = 0
        self.input_tokens = 0       # cumulative (for summary printout)
        self.output_tokens = 0
        self.last_input_tokens = 0  # most recent single LLM call (for live display)
        self.last_output_tokens = 0

    def start(self):
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def format_elapsed(self) -> str:
        total = int(self.elapsed)
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def record_tokens(self, message):
        """Extract token usage from a LangChain AIMessage.

        Updates cumulative totals (input_tokens, output_tokens) for the
        end-of-run summary, and last_* fields for the live per-call display.

        Tries usage_metadata first (TypedDict — dict access required),
        then response_metadata["token_usage"] as Ollama/OpenAI fallback.
        """
        inp = out = 0
        try:
            um = getattr(message, "usage_metadata", None)
            if um and isinstance(um, dict):
                inp = um.get("input_tokens") or 0
                out = um.get("output_tokens") or 0

            if not (inp or out):
                rm = getattr(message, "response_metadata", None)
                if rm and isinstance(rm, dict):
                    tu = rm.get("token_usage") or rm.get("usage") or {}
                    if isinstance(tu, dict):
                        inp = tu.get("prompt_tokens", 0) or 0
                        out = tu.get("completion_tokens", 0) or 0
        except Exception:
            pass

        if inp or out:
            self.input_tokens += inp
            self.output_tokens += out
            self.last_input_tokens = inp
            self.last_output_tokens = out

    def format_tokens(self, count: int) -> str:
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        if count >= 1_000:
            return f"{count / 1_000:.1f}K"
        return str(count)

    def record_write(self, target_path: str, new_content: str):
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    self.lines_deleted += len(f.read().splitlines())
            except Exception:
                pass
            self.files_overwritten += 1
        else:
            self.files_created += 1
        self.lines_written += len(new_content.splitlines())

    def print_summary(self, final_state: dict):
        history = final_state.get("history", [])
        plan = final_state.get("plan")
        completed = plan.completed_tasks if plan else []

        def _count(name):
            return history.count(name), history.count(f"{name}_failed")

        inspector_ok,  inspector_fail  = _count("inspector")
        architect_ok,  architect_fail  = _count("architect")
        coder_ok,      coder_fail      = _count("coder")
        validator_ok,  validator_fail  = _count("validator")

        net = self.lines_written - self.lines_deleted
        sign = "+" if net >= 0 else ""

        print()
        print("=" * 52)
        print("  WORKFLOW STATS")
        print("=" * 52)
        print(f"  Time elapsed:      {self.format_elapsed()}")
        print(f"  Iterations:        {final_state.get('iteration_count', 0)}")
        print(f"  Tasks completed:   {len(completed)}")
        print(f"  Coder retries:     {final_state.get('coder_retries', 0)}")
        print("-" * 52)
        print(f"  Inspector:         {inspector_ok} ok, {inspector_fail} failed")
        print(f"  Architect:         {architect_ok} ok, {architect_fail} failed")
        print(f"  Coder:             {coder_ok} ok, {coder_fail} failed")
        print(f"  Validator:         {validator_ok} ok, {validator_fail} failed")
        print("-" * 52)
        print(f"  Files created:     {self.files_created}")
        print(f"  Files overwritten: {self.files_overwritten}")
        print(f"  Lines written:     {self.lines_written}")
        print(f"  Lines deleted:     {self.lines_deleted}")
        print(f"  Net lines:         {sign}{net}")
        print("-" * 52)
        print(f"  Input tokens:      {self.format_tokens(self.input_tokens)}")
        print(f"  Output tokens:     {self.format_tokens(self.output_tokens)}")
        print(f"  Total tokens:      {self.format_tokens(self.total_tokens)}")

        if completed:
            print("-" * 52)
            print("  Completed tasks:")
            for i, task in enumerate(completed, 1):
                print(f"    {i}. {task}")

        print("=" * 52)


stats = WorkflowStats()
