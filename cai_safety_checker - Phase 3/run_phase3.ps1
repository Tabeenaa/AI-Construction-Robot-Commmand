# ─────────────────────────────────────────────────────────────────────────────
# run_phase3.ps1  —  Launch Phase 3 safety_checker / robot_executor
# ─────────────────────────────────────────────────────────────────────────────

$PYTHON = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  CAI Safety Checker — Phase 3                   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [1] safety_checker.py  (interactive REPL)"
Write-Host "  [2] robot_executor.py  (full simulation loop)"
Write-Host "  [3] run_verification.py (run test suite)"
Write-Host ""

$choice = Read-Host "Select (1/2/3)"

switch ($choice) {
    "1" { & $PYTHON (Join-Path $PSScriptRoot "safety_checker.py") @args }
    "2" { & $PYTHON (Join-Path $PSScriptRoot "robot_executor.py") @args }
    "3" { & $PYTHON (Join-Path $PSScriptRoot "run_verification.py") @args }
    default { Write-Host "Invalid selection" -ForegroundColor Red }
}
