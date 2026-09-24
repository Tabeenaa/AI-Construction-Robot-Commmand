import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "logs" / "safety_audit.db"

def init_db():
    """Ensure the logs directory and SQLite database exist with the proper schema."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            raw_command TEXT NOT NULL, -- Store raw command as JSON text
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            modified_command TEXT, -- Store modified command as JSON text or NULL
            principle_violated TEXT, -- Store list of violated principles as JSON text
            inference_time_ms REAL NOT NULL,
            tokens_generated INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_decision(timestamp: str, raw_command: str, decision: str, reason: str, 
                 modified_command: str | None, principle_violated: list[str], 
                 inference_time_ms: float, tokens_generated: int):
    """Log a decision row to the database."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Store raw_command as JSON text
    raw_command_json = json.dumps(raw_command)
    
    # Store modified_command as JSON text or NULL
    modified_command_json = json.dumps(modified_command) if modified_command is not None else None
    
    # Store principle_violated as JSON text list
    principle_violated_json = json.dumps(principle_violated)
    
    cursor.execute("""
        INSERT INTO decisions (
            timestamp, raw_command, decision, reason, 
            modified_command, principle_violated, 
            inference_time_ms, tokens_generated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        raw_command_json,
        decision,
        reason,
        modified_command_json,
        principle_violated_json,
        inference_time_ms,
        tokens_generated
    ))
    conn.commit()
    conn.close()

def get_decisions(limit=None, decision_filter=None, principle_filter=None):
    """Retrieve decision records from the database with optional filters."""
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
            raw_command = json.loads(row["raw_command"])
        except Exception:
            raw_command = row["raw_command"]
            
        try:
            modified_command = json.loads(row["modified_command"]) if row["modified_command"] is not None else None
        except Exception:
            modified_command = row["modified_command"]
            
        try:
            principle_violated = json.loads(row["principle_violated"]) if row["principle_violated"] is not None else []
        except Exception:
            principle_violated = [row["principle_violated"]] if row["principle_violated"] else []
            
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "raw_command": raw_command,
            "decision": row["decision"],
            "reason": row["reason"],
            "modified_command": modified_command,
            "principle_violated": principle_violated,
            "inference_time_ms": row["inference_time_ms"],
            "tokens_generated": row["tokens_generated"]
        })
        
    conn.close()
    return results

def get_stats():
    """Calculate statistics across logged decisions."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total FROM decisions")
    total = cursor.fetchone()["total"]
    
    if total == 0:
        conn.close()
        return {
            "total": 0,
            "refuse_count": 0,
            "refuse_rate_pct": 0.0,
            "avg_inference_time_ms": 0.0,
            "max_inference_time_ms": 0.0,
            "total_tokens": 0
        }
        
    cursor.execute("SELECT COUNT(*) as refuse_count FROM decisions WHERE decision = 'REFUSE'")
    refuse_count = cursor.fetchone()["refuse_count"]
    
    cursor.execute("SELECT AVG(inference_time_ms) as avg_time, MAX(inference_time_ms) as max_time FROM decisions")
    time_row = cursor.fetchone()
    avg_time = time_row["avg_time"] or 0.0
    max_time = time_row["max_time"] or 0.0
    
    cursor.execute("SELECT SUM(tokens_generated) as total_tokens FROM decisions")
    total_tokens = cursor.fetchone()["total_tokens"] or 0
    
    conn.close()
    
    return {
        "total": total,
        "refuse_count": refuse_count,
        "refuse_rate_pct": round((refuse_count / total) * 100, 1),
        "avg_inference_time_ms": round(avg_time, 1),
        "max_inference_time_ms": round(max_time, 1),
        "total_tokens": total_tokens
    }

def export_to_json(filepath: str, limit=None, decision_filter=None, principle_filter=None) -> int:
    """Export queried decisions to a JSON file. Returns count of exported rows."""
    decisions = get_decisions(limit=limit, decision_filter=decision_filter, principle_filter=principle_filter)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "export_timestamp": datetime.now().isoformat(),
            "count": len(decisions),
            "decisions": decisions
        }, f, indent=2)
    return len(decisions)
