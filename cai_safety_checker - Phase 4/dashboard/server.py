"""
dashboard/server.py
===================
Lightweight HTTP & API server for the Construction Robot Web Command Center.
Serves the elegant light-themed dashboard and provides real-time audit & telemetry endpoints.
"""

import sys
import os
import json
import http.server
import socketserver
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add parent directory for imports
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR))

from db import get_decisions, get_stats, log_decision
from safety_checker import RobotSafetyChecker

PORT = 8080
DASHBOARD_DIR = Path(__file__).parent

# Cache a checker instance
checker = None

def get_checker():
    global checker
    if checker is None:
        checker = RobotSafetyChecker()
    return checker


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        
        # API: /api/stats
        if parsed.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            stats = get_stats()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return
            
        # API: /api/decisions
        elif parsed.path == "/api/decisions":
            query_params = parse_qs(parsed.query)
            limit = int(query_params.get("limit", [25])[0])
            filter_type = query_params.get("filter", [None])[0]
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            decisions = get_decisions(limit=limit, decision_filter=filter_type)
            self.wfile.write(json.dumps(decisions).encode("utf-8"))
            return
            
        # Fallback to static file server
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        
        # API: /api/audit_command
        if parsed.path == "/api/audit_command":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            print(f"[API] Received audit request: {body[:60]}...", flush=True)
            
            try:
                payload = json.loads(body)
                cmd_text = payload.get("command", "").strip()
                if not cmd_text:
                    self.send_error(400, "Missing command string")
                    return

                # Audit command
                c = get_checker()
                verdict = c.check(cmd_text)
                print(f"[API] Verdict: {verdict.verdict} ({verdict.inference_time_ms:.1f}ms)", flush=True)
                
                resp_data = {
                    "command": verdict.command,
                    "verdict": verdict.verdict,
                    "violated_principles": verdict.violated_principles,
                    "reason": verdict.reason,
                    "modified_command": verdict.modified_command,
                    "confidence": verdict.confidence,
                    "extracted_parameters": verdict.extracted_parameters,
                    "inference_time_ms": verdict.inference_time_ms,
                    "timestamp": verdict.timestamp
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(resp_data).encode("utf-8"))
                return
            except Exception as e:
                self.send_error(500, f"Error processing command: {e}")
                return

        self.send_error(404, "Endpoint not found")


def run_server():
    print(f"\n=======================================================", flush=True)
    print(f"  AI CONSTRUCTION ROBOT COMMAND CENTER (LIGHT THEME)  ", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  Server listening on: http://localhost:{PORT}", flush=True)
    print(f"  Access the dashboard in your web browser.\n", flush=True)
    
    # Preload checker
    get_checker()
    
    # Threading server
    with http.server.ThreadingHTTPServer(("", PORT), DashboardRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.", flush=True)


if __name__ == "__main__":
    run_server()
