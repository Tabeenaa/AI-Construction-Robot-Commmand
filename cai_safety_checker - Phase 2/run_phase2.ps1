# ─────────────────────────────────────────────────────────────────────────────
# run_phase2.ps1  —  Launch Phase 2 safety_checker using the shared venv
#
# Phase 2's own venv is missing llama-cpp-python.
# Phase 3's venv has llama_cpp_python 0.2.90 + all dependencies.
# This script uses Phase 3's Python to run Phase 2 code.
# ─────────────────────────────────────────────────────────────────────────────

$PYTHON = Join-Path $PSScriptRoot "..\cai_safety_checker - Phase 3\venv\Scripts\python.exe"
$SCRIPT  = Join-Path $PSScriptRoot "safety_checker.py"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  CAI Safety Checker — Phase 2 (shared venv)     " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Python  : $PYTHON" -ForegroundColor Gray
Write-Host "  Script  : $SCRIPT" -ForegroundColor Gray
Write-Host ""

# Pass any arguments through (e.g. a command string or --test)
& $PYTHON $SCRIPT @args
