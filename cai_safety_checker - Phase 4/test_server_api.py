import urllib.request
import json

url = "http://localhost:8080/api/audit_command"
data = {"command": "drill anchor hole at X=0.45, Y=0.20, Z=0.35 at safe speed 0.30 m/s with worker at 1.2 m"}
req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("\n>>> POST /api/audit_command Response: <<<")
        print(json.dumps(res, indent=2))
except Exception as e:
    print(f"Error: {e}")
