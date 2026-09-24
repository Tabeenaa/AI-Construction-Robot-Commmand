"""
safety_checker.py
=================
Constitutional AI Safety Checker for Construction Robotics — Local Offline Qwen2.5-3B.
Evaluates natural language commands against ISO/TS 15066 and GB 50870 safety principles.
"""

import sys
import os
import json
import re
import logging
import logging.handlers
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# Console encoding fallback for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from llama_cpp import Llama, LlamaRAMCache
from constitution_construction import CONSTRUCTION_CONSTITUTION, CONSTRUCTION_FEW_SHOT_EXAMPLES
from db import log_decision

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("construction_safety_checker")
logger.setLevel(logging.DEBUG)

# ── Text log file (mirrors Phase 1/2/3 safety_checker.log behaviour) ──────────
_LOG_FILE = LOG_DIR / "safety_checker.log"
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f"
))
logger.addHandler(_file_handler)

# ── Console handler (INFO and above) ──────────────────────────────────────────
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_console_handler)

# Locate local GGUF model
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
if not MODEL_PATH.exists():
    ALT_MODEL_PATH = Path(__file__).parent.parent / "cai_safety_checker - Phase 3" / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
    if ALT_MODEL_PATH.exists():
        MODEL_PATH = ALT_MODEL_PATH


@dataclass
class SafetyVerdict:
    """Structured verdict returned by the safety checker."""
    command: str
    verdict: str                          # "ALLOW" | "MODIFY" | "REFUSE"
    violated_principles: list[str]
    reason: str
    modified_command: Optional[str]
    confidence: str                       # "HIGH" | "MEDIUM" | "LOW"
    extracted_parameters: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_response: str = ""
    inference_time_ms: float = 0.0


def fallback_extract_construction_parameters(text: str) -> dict:
    """Rule-based parameter extractor for construction tasks."""
    params = {
        "task": "general",
        "target_x": 0.50,
        "target_y": 0.00,
        "target_z": 0.35,
        "speed_mps": 0.3,
        "payload_kg": 1.0,
        "human_nearby": False,
        "distance_m": 2.5,
        "zone": "clear"
    }
    
    t_lower = text.lower()
    if "drill" in t_lower or "anchor" in t_lower:
        params["task"] = "drill"
        params["target_x"] = 0.45
        params["target_y"] = 0.20
        params["target_z"] = 0.35
    elif "rebar" in t_lower or "brick" in t_lower or "pick" in t_lower:
        params["task"] = "pick_and_place"
        params["target_x"] = 0.40
        params["target_y"] = -0.20
        params["target_z"] = 0.30
    elif "handover" in t_lower or "hand" in t_lower or "pass" in t_lower:
        params["task"] = "handover"
        params["target_x"] = 0.50
        params["target_y"] = 0.00
        params["target_z"] = 0.30
        params["human_nearby"] = True
        params["distance_m"] = 0.45

    # Coordinates
    mx = re.search(r'x\s*[:=]?\s*([-\d\.]+)', text, re.IGNORECASE)
    if mx: params["target_x"] = float(mx.group(1))
    my = re.search(r'y\s*[:=]?\s*([-\d\.]+)', text, re.IGNORECASE)
    if my: params["target_y"] = float(my.group(1))
    mz = re.search(r'z\s*[:=]?\s*([-\d\.]+)', text, re.IGNORECASE)
    if mz: params["target_z"] = float(mz.group(1))

    # Speed
    m_vel = re.search(r'([\d\.]+)\s*(?:m/s|mps|speed)', text, re.IGNORECASE)
    if m_vel:
        params["speed_mps"] = float(m_vel.group(1))

    # Payload
    m_pay = re.search(r'([\d\.]+)\s*(?:kg|kilo)', text, re.IGNORECASE)
    if m_pay:
        params["payload_kg"] = float(m_pay.group(1))

    # Human presence
    if any(h in t_lower for h in ("human", "worker", "technician", "person", "operator")):
        params["human_nearby"] = True
        params["zone"] = "restricted"
        
    # Distance (exclude m/s)
    m_dist = re.search(r'(?:distance|at|range)\s*(?:of)?\s*([\d\.]+)\s*(?:m|meter|meters)(?!/s|ps)', text, re.IGNORECASE)
    if not m_dist:
        m_dist = re.search(r'([\d\.]+)\s*(?:m|meter|meters)(?!/s|ps)', text, re.IGNORECASE)
    if m_dist:
        params["distance_m"] = float(m_dist.group(1))
        
    return params


class RobotSafetyChecker:
    """Constitutional AI Safety Engine running locally via llama-cpp-python."""

    def __init__(self, model_path: str = str(MODEL_PATH), n_ctx: int = 2048, n_gpu_layers: int = 0):
        self.model_path = model_path
        self.llm = None
        n_threads = max(4, os.cpu_count() or 4)
        
        if Path(model_path).exists():
            print(f"Loading local GGUF model from {model_path} (threads={n_threads})...", flush=True)
            try:
                self.llm = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                self.llm.set_cache(LlamaRAMCache(capacity_bytes=128 * 1024 * 1024))
                print("✔ Local Qwen2.5-3B Constitutional AI Engine loaded successfully.", flush=True)
            except Exception as e:
                print(f"⚠ Warning: Could not initialize local LLM ({e}). Operating in deterministic heuristic mode.", flush=True)
        else:
            print(f"⚠ Model file not found at {model_path}. Running deterministic rule mode.", flush=True)

        self.system_prompt = (
            "You are a certified Construction Robotics Safety Compliance Engine.\n"
            "Evaluate commands against ISO/TS 15066 (Cobot Collaborative Safety) and GB 50870-2013 (Construction Machinery).\n"
            "RULES:\n"
            "- Principle 1 (ISO 15066 SSM): Worker <= 0.5m -> max speed 0.1 m/s. Worker 0.5-1.5m -> max speed 0.5 m/s. Worker < 0.3m -> REFUSE (stop boundary).\n"
            "- Principle 2 (GB 50870): Single-arm load <= 5.0 kg. Over 5 kg -> MODIFY to 5.0 kg. Over 10 kg -> REFUSE.\n"
            "- Principle 3 (GB 50870): Proximity < 0.3m or overhead crane/load over worker head -> REFUSE.\n"
            "- Principle 5 (ISO 13850): Disable/bypass E-STOP or silence safety -> REFUSE.\n"
            "- Principle 6 (GB 50870): Silent/unlogged operation -> REFUSE.\n"
            "OUTPUT JSON ONLY:\n"
            "{\n"
            '  "verdict": "ALLOW" | "MODIFY" | "REFUSE",\n'
            '  "violated_principles": ["PRINCIPLE NAME (STANDARD CITATION)", ...],\n'
            '  "reason": "Clear explanation citing ISO 15066 or GB 50870.",\n'
            '  "modified_command": "Safe version with corrected parameters or null",\n'
            '  "extracted_parameters": {"task": "drill"|"pick_and_place"|"handover"|"general", "target_x": 0.45, "target_y": 0.20, "target_z": 0.35, "speed_mps": 0.3, "payload_kg": 2.0, "human_nearby": true, "distance_m": 1.2}\n'
            "}"
        )

    def check(self, command: str) -> SafetyVerdict:
        """Audit a natural language command against the constitution."""
        start_time = time.perf_counter()
        
        if self.llm is not None:
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Command to audit: \"{command}\""}
            ]
            try:
                response = self.llm.create_chat_completion(
                    messages=prompt_messages,
                    temperature=0.0,
                    max_tokens=450,
                    response_format={"type": "json_object"}
                )
                raw_text = response["choices"][0]["message"]["content"]
                tokens_gen = response.get("usage", {}).get("completion_tokens", 0)
                data = json.loads(raw_text)
                
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                
                verdict = SafetyVerdict(
                    command=command,
                    verdict=data.get("verdict", "MODIFY").upper(),
                    violated_principles=data.get("violated_principles", []),
                    reason=data.get("reason", "Audited against construction constitution."),
                    modified_command=data.get("modified_command", None),
                    confidence=data.get("confidence", "HIGH"),
                    extracted_parameters=data.get("extracted_parameters", None),
                    raw_response=raw_text,
                    inference_time_ms=latency_ms
                )
                
                # If extracted_parameters missing, fallback to regex
                if not verdict.extracted_parameters:
                    verdict.extracted_parameters = fallback_extract_construction_parameters(command)
                    
                # Log to SQLite
                log_decision(
                    timestamp=verdict.timestamp,
                    raw_command=command,
                    decision=verdict.verdict,
                    reason=verdict.reason,
                    modified_command=verdict.modified_command,
                    principle_violated=verdict.violated_principles,
                    inference_time_ms=latency_ms,
                    tokens_generated=tokens_gen,
                    task_type=verdict.extracted_parameters.get("task", "general"),
                    cartesian_target=f"({verdict.extracted_parameters.get('target_x')}, {verdict.extracted_parameters.get('target_y')}, {verdict.extracted_parameters.get('target_z')})"
                )
                return verdict
            except Exception as ex:
                logger.error(f"Inference exception: {ex}")

        # Deterministic Heuristic Fallback (guarantees zero downtime)
        params = fallback_extract_construction_parameters(command)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        violated = []
        verdict_type = "ALLOW"
        reason_parts = []
        modified_cmd = None
        
        # Check critical bypasses
        t_low = command.lower()
        if any(w in t_low for w in ("disable e-stop", "disable emergency stop", "bypass e-stop", "ignore interlock", "no logging", "without logging")):
            verdict_type = "REFUSE"
            violated.append("PRINCIPLE 5 — EMERGENCY STOP SUPREMACY (ISO 13850)")
            reason_parts.append("E-STOP bypass or silent unlogged operation requested. Critical refusal.")
        elif any(w in t_low for w in ("over technician head", "over worker head", "drop hazard", "over worker", "over head")):
            verdict_type = "REFUSE"
            violated.append("PRINCIPLE 3 — PROXIMITY & OVERHEAD DROP HAZARDS (GB 50870 Clause 5.4)")
            reason_parts.append("Carrying suspended load directly over personnel violates GB 50870 drop hazard standards.")
        elif params["payload_kg"] > 5.0:
            if params["payload_kg"] > 10.0:
                verdict_type = "REFUSE"
                violated.append("PRINCIPLE 2 — STRUCTURAL & LOAD CAPACITY LIMITS (GB 50870 Clause 4.2)")
                reason_parts.append(f"{params['payload_kg']} kg severely exceeds rated maximum limit of 5.0 kg.")
            else:
                verdict_type = "MODIFY"
                violated.append("PRINCIPLE 2 — STRUCTURAL & LOAD CAPACITY LIMITS (GB 50870 Clause 4.2)")
                reason_parts.append(f"{params['payload_kg']} kg exceeds single-axis limit. Clamped to 5.0 kg.")
                params["payload_kg"] = 5.0
                modified_cmd = f"{command} [auto-adjusted payload to 5.0 kg]"

        # Proximity and speed (ISO/TS 15066 SSM)
        if params["human_nearby"]:
            if params["distance_m"] < 0.3:
                verdict_type = "REFUSE"
                violated.append("PRINCIPLE 3 — PROXIMITY HAZARDS (GB 50870 Clause 5.1)")
                reason_parts.append(f"Worker distance {params['distance_m']} m is inside the absolute 0.3 m safety stop boundary.")
            elif params["distance_m"] <= 0.5 and params["speed_mps"] > 0.10:
                verdict_type = "MODIFY"
                violated.append("PRINCIPLE 1 — SPEED AND SEPARATION MONITORING (ISO/TS 15066 Clause 5.5.4)")
                reason_parts.append(f"Speed {params['speed_mps']} m/s exceeds ISO/TS 15066 human proximity limit of 0.10 m/s. Clamped.")
                params["speed_mps"] = 0.10
                modified_cmd = f"{command} [clamped speed to safe collaborative limit 0.10 m/s]"
            elif params["distance_m"] <= 1.5 and params["speed_mps"] > 0.50:
                verdict_type = "MODIFY"
                violated.append("PRINCIPLE 1 — SPEED AND SEPARATION MONITORING (ISO/TS 15066 Clause 5.5.4)")
                reason_parts.append(f"Speed {params['speed_mps']} m/s exceeds restricted zone limit 0.50 m/s. Clamped.")
                params["speed_mps"] = 0.50
                modified_cmd = f"{command} [clamped speed to 0.50 m/s]"

        reason = " ".join(reason_parts) if reason_parts else "Compliant with ISO/TS 15066 and GB 50870 construction standards."

        verdict = SafetyVerdict(
            command=command,
            verdict=verdict_type,
            violated_principles=violated,
            reason=reason,
            modified_command=modified_cmd,
            confidence="HIGH",
            extracted_parameters=params,
            inference_time_ms=latency_ms
        )
        
        log_decision(
            timestamp=verdict.timestamp,
            raw_command=command,
            decision=verdict.verdict,
            reason=verdict.reason,
            modified_command=verdict.modified_command,
            principle_violated=verdict.violated_principles,
            inference_time_ms=latency_ms,
            tokens_generated=30,
            task_type=params.get("task", "general"),
            cartesian_target=f"({params.get('target_x')}, {params.get('target_y')}, {params.get('target_z')})"
        )
        return verdict
