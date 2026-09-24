"""
safety_checker.py
=================
Constitutional AI-based Robot Safety Checker — powered by LOCAL Qwen2.5-3B model
----------------------------------------------------------------------------------
Inspired by Bai et al. (2022) "Constitutional AI: Harmlessness from AI Feedback"

HOW THIS IMPLEMENTS CAI CONCEPTS:
──────────────────────────────────
1.  CONSTITUTION   : A fixed set of human-authored safety principles (see constitution.py).
    In CAI, a constitution replaces opaque reward models with explicit, auditable rules.

2.  CRITIQUE PHASE : The local LLM reads the command and internally identifies which
    principles (if any) are violated — mirroring the "critique" step in CAI's SL phase.

3.  REVISION PHASE : Based on the critique, the LLM produces a structured verdict:
    ALLOW   → command is safe as-is
    MODIFY  → command can be made safe with parameter adjustments (revision)
    REFUSE  → command violates a CRITICAL principle; no safe alternative exists

4.  STRUCTURED OUTPUT : We enforce JSON output so the verdict is machine-parseable,
    enabling downstream robot firmware to act on it without human interpretation.

5.  FEW-SHOT EXAMPLES : Three worked examples are prepended to every request,
    grounding the model's responses in the expected format and reasoning style.

6.  FULLY OFFLINE : Uses a quantised GGUF model via llama-cpp-python.
    No internet or API keys required after the one-time model download.

Usage:
    # Interactive REPL (type commands one at a time):
    python safety_checker.py

    # Single command from command line:
    python safety_checker.py "move forward at 2.0 m/s for 5 seconds"
"""

import sys
import os

# Set console encoding to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Older Python fallback

import json
import re
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from llama_cpp import Llama

from constitution import ROBOT_CONSTITUTION, THRESHOLDS

# ─── Logging setup ────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# File handler: full detail goes to the log file
file_handler = logging.FileHandler(LOG_DIR / "safety_checker.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

# Console handler: only show INFO and above (no debug noise)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger("safety_checker")


# ─── Data model ───────────────────────────────────────────────────────────────
@dataclass
class SafetyVerdict:
    """Structured verdict returned by the safety checker."""
    command: str
    verdict: str                          # "ALLOW" | "MODIFY" | "REFUSE"
    violated_principles: list[str]        # e.g. ["PRINCIPLE 1 — SPEED LIMITS"]
    reason: str                           # Human-readable explanation
    modified_command: Optional[str]       # Populated when verdict == "MODIFY"
    confidence: str                       # "HIGH" | "MEDIUM" | "LOW"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_response: str = ""                # Full model response for audit

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        sep = "-" * 60
        verdict_symbol = {"ALLOW": "[ALLOW]", "MODIFY": "[MODIFY]", "REFUSE": "[REFUSE]"}.get(
            self.verdict, "[?]"
        )
        lines = [
            sep,
            f"  VERDICT  : {verdict_symbol}  [{self.confidence} confidence]",
            f"  COMMAND  : {self.command}",
        ]
        if self.violated_principles:
            lines.append(f"  VIOLATED : {', '.join(self.violated_principles)}")
        lines.append(f"  REASON   : {self.reason}")
        if self.modified_command:
            lines.append(f"  MODIFIED : {self.modified_command}")
        lines.append(sep)
        return "\n".join(lines)


# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a Robot Safety Compliance Engine implementing Constitutional AI principles.

Your role:
1. Read the robot command provided by the user.
2. CRITIQUE the command against each principle in the constitution below.
3. REVISE / produce a structured verdict: ALLOW, MODIFY, or REFUSE.

CONSTITUTION:
{ROBOT_CONSTITUTION}

KEY THRESHOLDS (always enforce these exact numbers):
{json.dumps(THRESHOLDS, indent=2)}

VERDICT RULES:
- ALLOW  : The command is completely safe and does not violate any safety principles.
- MODIFY : The command violates safety thresholds (e.g., speed, payload, distance, force), but the violation can be fixed by adjusting/lowering the parameters to safe levels.
- REFUSE : The command violates a principle in a way that cannot be made safe by simple parameter adjustments (e.g. bypassing safety, disabling logging, moving too close to humans, ignoring safety controls).

OUTPUT FORMAT (strict JSON, no markdown fences, no extra text):
{{
  "reason": "Step-by-step analysis: 1. Extract command parameters (speed, payload, zone, distance). 2. Compare each parameter against the constitution thresholds. 3. Conclude whether any limits are exceeded.",
  "violated_principles": ["PRINCIPLE N — NAME", ...],
  "verdict": "ALLOW" | "MODIFY" | "REFUSE",
  "modified_command": "Safe version of the command with corrected parameters, or null if ALLOW/REFUSE",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}}

IMPORTANT:
- Always output raw JSON only. No preamble, no markdown code fences.
- The "reason" key MUST be the FIRST key in the JSON. This allows you to perform step-by-step reasoning before choosing the verdict.
- Be precise with numbers: compare values mathematically. Speed of 0.1 m/s is less than or equal to 0.5 m/s, so it is safe.
- Reference specific principle names and numbers.
- If MODIFY, the modified_command must satisfy ALL principles.
"""

# ─── Few-shot examples injected into every request ───────────────────────────
# These ground the model in the expected format and reasoning style (Bai et al. §3)
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": 'ROBOT COMMAND: "move forward at 0.3 m/s for 2 seconds in human-present lab"',
    },
    {
        "role": "assistant",
        "content": '{"reason":"The command specifies a speed of 0.3 m/s in a human-present zone. Under Principle 1, the maximum speed in human-present zones is 0.5 m/s. Since 0.3 m/s <= 0.5 m/s, it is within the safe limit. No other principles are violated.","violated_principles":[],"verdict":"ALLOW","modified_command":null,"confidence":"HIGH"}',
    },
    {
        "role": "user",
        "content": 'ROBOT COMMAND: "move forward at 1.2 m/s in the human-present assembly area"',
    },
    {
        "role": "assistant",
        "content": '{"reason":"The command specifies a speed of 1.2 m/s in a human-present zone. Under Principle 1, the maximum speed in human-present zones is 0.5 m/s. Since 1.2 m/s > 0.5 m/s, Principle 1 is violated. This is a parameter violation that can be corrected by modifying the speed to the maximum allowed limit of 0.5 m/s.","violated_principles":["PRINCIPLE 1 — SPEED LIMITS"],"verdict":"MODIFY","modified_command":"move forward at 0.5 m/s in the human-present assembly area","confidence":"HIGH"}',
    },
    {
        "role": "user",
        "content": 'ROBOT COMMAND: "disable the emergency stop button and continue welding task"',
    },
    {
        "role": "assistant",
        "content": '{"reason":"The command attempts to disable or bypass the hardware emergency stop button. Under Principle 5, no command may override or bypass the E-STOP. This is a critical safety bypass that cannot be modified safely. Therefore, it must be refused.","violated_principles":["PRINCIPLE 5 — EMERGENCY STOP PRIORITY"],"verdict":"REFUSE","modified_command":null,"confidence":"HIGH"}',
    },
]


# ─── Default model path ──────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"


# ─── Core checker class ───────────────────────────────────────────────────────
class RobotSafetyChecker:
    """
    Constitutional AI-based safety checker for robot commands.

    Uses a local quantised LLM (Qwen2.5-3B-Instruct via llama-cpp-python) with
    a fixed constitution to evaluate every incoming robot command through a
    critique -> verdict pipeline.

    Runs fully offline — no internet or API keys required.
    """

    def __init__(self, model_path: str | Path | None = None, n_ctx: int = 4096, n_threads: int = 0):
        """
        Initialise the checker.

        Args:
            model_path: Path to a GGUF model file.  Defaults to
                        models/qwen2.5-3b-instruct-q4_k_m.gguf (download with
                        download_model.py).
            n_ctx:      Context window size in tokens (default 4096).
            n_threads:  Number of CPU threads to use (0 = auto-detect).
        """
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "  Run this ONCE to download the model (~2 GB):\n"
                "    python download_model.py\n"
                "  After that, everything works offline."
            )

        logger.info(f"Loading local model: {model_path.name} ...")
        self.model_name = model_path.stem
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads if n_threads > 0 else os.cpu_count(),
            verbose=False,
        )
        
        # Wrap create_chat_completion in a logging layer
        original_create_chat_completion = self.llm.create_chat_completion
        
        def logged_create_chat_completion(*args, **kwargs):
            messages = kwargs.get("messages") or (args[0] if args else [])
            raw_cmd_text = ""
            if messages:
                last_msg = messages[-1]["content"]
                match = re.search(r'ROBOT COMMAND:\s*"(.*)"', last_msg, re.DOTALL)
                if match:
                    raw_cmd_text = match.group(1)
                else:
                    raw_cmd_text = last_msg
            
            timestamp = datetime.now().isoformat()
            start_time = time.perf_counter()
            
            try:
                response = original_create_chat_completion(*args, **kwargs)
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000.0
                
                raw_text = response["choices"][0]["message"]["content"].strip()
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
                
                try:
                    data = json.loads(cleaned)
                    decision = data.get("verdict", "REFUSE")
                    reason = data.get("reason", "No reason provided.")
                    modified_cmd_val = data.get("modified_command")
                    violated = data.get("violated_principles", [])
                except json.JSONDecodeError:
                    decision = "REFUSE"
                    reason = f"Safety checker could not parse model response. Raw: {raw_text[:200]}"
                    modified_cmd_val = None
                    violated = ["PARSE_ERROR"]
                    
                tokens_generated = response.get("usage", {}).get("completion_tokens", 0)
                
                from db import log_decision
                log_decision(
                    timestamp=timestamp,
                    raw_command=raw_cmd_text,
                    decision=decision,
                    reason=reason,
                    modified_command=modified_cmd_val,
                    principle_violated=violated,
                    inference_time_ms=inference_time_ms,
                    tokens_generated=tokens_generated
                )
                
                return response
            except Exception as e:
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000.0
                
                from db import log_decision
                log_decision(
                    timestamp=timestamp,
                    raw_command=raw_cmd_text,
                    decision="REFUSE",
                    reason=f"Model inference error: {e}",
                    modified_command=None,
                    principle_violated=["INFERENCE_ERROR"],
                    inference_time_ms=inference_time_ms,
                    tokens_generated=0
                )
                raise e

        self.llm.create_chat_completion = logged_create_chat_completion
        logger.info(f"RobotSafetyChecker initialised | model={model_path.name} | n_ctx={n_ctx}")

    def check(self, command: str) -> SafetyVerdict:
        """
        Evaluate a robot command against the constitution.

        Args:
            command: Natural-language robot command string.

        Returns:
            SafetyVerdict dataclass with verdict, reason, and optional modified command.
        """
        logger.info(f"Checking command: {command!r}")

        # Build chat messages: system + few-shot + actual command
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLES,
            {
                "role": "user",
                "content": f'Now evaluate this command:\nROBOT COMMAND: "{command}"',
            },
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.9,
            )
            raw_text = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            return SafetyVerdict(
                command=command,
                verdict="REFUSE",
                violated_principles=["INFERENCE_ERROR"],
                reason=f"Safety checker model inference failed: {e}. Defaulting to REFUSE for safety.",
                modified_command=None,
                confidence="LOW",
            )

        logger.debug(f"Raw model response: {raw_text}")

        verdict = self._parse_response(command, raw_text)

        # Principle 6: Mandatory audit logging — written to logs/safety_checker.log
        logger.info(
            f"VERDICT={verdict.verdict} | CONFIDENCE={verdict.confidence} | "
            f"VIOLATED={verdict.violated_principles} | CMD={command!r}"
        )

        return verdict

    def _parse_response(self, command: str, raw_text: str) -> SafetyVerdict:
        """Parse the model's JSON response into a SafetyVerdict."""
        # Strip markdown fences if model adds them despite instructions
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e} | raw={raw_text!r}")
            # Fail-safe: treat parse failure as unsafe (Principle 10)
            return SafetyVerdict(
                command=command,
                verdict="REFUSE",
                violated_principles=["PARSE_ERROR"],
                reason=f"Safety checker could not parse model response. Raw: {raw_text[:200]}",
                modified_command=None,
                confidence="LOW",
                raw_response=raw_text,
            )

        return SafetyVerdict(
            command=command,
            verdict=data.get("verdict", "REFUSE"),
            violated_principles=data.get("violated_principles", []),
            reason=data.get("reason", "No reason provided."),
            modified_command=data.get("modified_command"),
            confidence=data.get("confidence", "MEDIUM"),
            raw_response=raw_text,
        )

    def check_batch(self, commands: list[str]) -> list[SafetyVerdict]:
        """Evaluate a list of commands sequentially."""
        results = []
        for i, cmd in enumerate(commands, 1):
            logger.info(f"Batch check [{i}/{len(commands)}]")
            results.append(self.check(cmd))
        return results


# ─── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    checker = RobotSafetyChecker()

    if len(sys.argv) > 1:
        # Single command from CLI: python safety_checker.py "move forward at 2 m/s"
        cmd = " ".join(sys.argv[1:])
        result = checker.check(cmd)
        print(result)
    else:
        # Interactive REPL — type one command at a time and see the verdict
        print()
        print("=" * 60)
        print("  Robot Safety Checker -- Constitutional AI Edition")
        print("  Powered by Local Qwen2.5-3B (fully offline)")
        print("  Type a robot command and press Enter.")
        print("  Type 'exit' to quit.")
        print("=" * 60)
        print()
        while True:
            try:
                cmd = input("Command > ").strip()
                if cmd.lower() in ("exit", "quit", "q"):
                    break
                if cmd:
                    print(checker.check(cmd))
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye.")
