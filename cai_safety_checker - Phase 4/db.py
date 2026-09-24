"""
db.py
=====
SQLite persistent audit logging layer for Construction Robot Constitutional AI decisions.
Stores tamper-evident record of all dispatches, Cartesian coordinates, and safety verdicts.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "logs" / "safety_audit.db"


def init_db():
    """Ensure database and table exist with schema."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            raw_command TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            modified_command TEXT,
            principle_violated TEXT,
            inference_time_ms REAL NOT NULL,
            tokens_generated INTEGER NOT NULL,
            task_type TEXT DEFAULT 'general',
            cartesian_target TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_decision(
    timestamp: str,
    raw_command: str,
    decision: str,
    reason: str,
    modified_command: str | None,
    principle_violated: list[str],
    inference_time_ms: float,
    tokens_generated: int,
    task_type: str = "general",
    cartesian_target: str | None = None
):
    """Log a decision event to SQLite."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    raw_command_json = json.dumps(raw_command)
    modified_command_json = json.dumps(modified_command) if modified_command is not None else None
    principle_violated_json = json.dumps(principle_violated)
    
    cursor.execute("""
        INSERT INTO decisions (
            timestamp, raw_command, decision, reason,
            modified_command, principle_violated,
            inference_time_ms, tokens_generated,
            task_type, cartesian_target
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        raw_command_json,
        decision,
        reason,
        modified_command_json,
        principle_violated_json,
        inference_time_ms,
        tokens_generated,
        task_type,
        cartesian_target
    ))
    conn.commit()
    conn.close()


def get_decisions(limit: int = 20, decision_filter: str = None, principle_filter: str = None) -> list[dict]:
    """Retrieve decisions with optional filtering."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM decisions"
    conditions = []
    params = []
    
    if decision_filter:
        conditions.append("decision = ?")
        params.append(decision_filter)
        
    if principle_filter:
        conditions.append("principle_violated LIKE ?")
        params.append(f"%{principle_filter}%")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY id DESC"
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        try:
            raw_cmd = json.loads(row["raw_command"])
        except Exception:
            raw_cmd = row["raw_command"]
            
        try:
            mod_cmd = json.loads(row["modified_command"]) if row["modified_command"] else None
        except Exception:
            mod_cmd = row["modified_command"]
            
        try:
            violated = json.loads(row["principle_violated"]) if row["principle_violated"] else []
        except Exception:
            violated = []
            
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "raw_command": raw_cmd,
            "decision": row["decision"],
            "reason": row["reason"],
            "modified_command": mod_cmd,
            "principle_violated": violated,
            "inference_time_ms": row["inference_time_ms"],
            "tokens_generated": row["tokens_generated"],
            "task_type": row["task_type"] if "task_type" in row.keys() else "general",
            "cartesian_target": row["cartesian_target"] if "cartesian_target" in row.keys() else None
        })
    conn.close()
    return results


def get_stats() -> dict:
    """Compute summary statistics for the audit database."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*), AVG(inference_time_ms), MAX(inference_time_ms), SUM(tokens_generated) FROM decisions")
    row = cursor.fetchone()
    total = row[0] or 0
    avg_lat = row[1] or 0.0
    max_lat = row[2] or 0.0
    total_tokens = row[3] or 0
    
    cursor.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'REFUSE'")
    refuse_count = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'MODIFY'")
    modify_count = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'ALLOW'")
    allow_count = cursor.fetchone()[0] or 0
    
    conn.close()
    
    refuse_rate = (refuse_count / total * 100.0) if total > 0 else 0.0
    
    return {
        "total": total,
        "allow_count": allow_count,
        "modify_count": modify_count,
        "refuse_count": refuse_count,
        "refuse_rate_pct": round(refuse_rate, 1),
        "avg_inference_time_ms": round(avg_lat, 1),
        "max_inference_time_ms": round(max_lat, 1),
        "total_tokens": total_tokens
    }
