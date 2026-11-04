$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
$Profile = if ($args.Count -gt 0) { $args[0] } else { "cpu" }
if ($Profile -notin @("cpu", "cuda", "dev")) {
    throw "usage: .\setup_avril.ps1 [cpu|cuda|dev]"
}
$Requirement = if ($Profile -eq "cuda") { "requirements/cuda12.txt" } elseif ($Profile -eq "dev") { "requirements/dev.txt" } else { "requirements/cpu.txt" }
$Python = if ($env:AVRIL_PYTHON) { $env:AVRIL_PYTHON } else { "python" }
$PipCache = if ($env:AVRIL_PIP_CACHE_DIR) {
    $env:AVRIL_PIP_CACHE_DIR
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "Avril\pip-cache"
} else {
    Join-Path $env:USERPROFILE ".cache\avril\pip"
}
New-Item -ItemType Directory -Force -Path $PipCache | Out-Null
& $Python -c "import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))"
if ($LASTEXITCODE -ne 0) {
    throw "Avril requires Python 3.11-3.13; Python 3.11 or 3.12 is recommended."
}
if (-not (Test-Path ".venv/Scripts/python.exe")) { & $Python -m venv .venv }
$PreviousPipCache = $env:PIP_CACHE_DIR
try {
    $env:PIP_CACHE_DIR = $PipCache
    & .venv/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
    & .venv/Scripts/python.exe -m pip install -r $Requirement
    & .venv/Scripts/python.exe -m pip install . --no-deps --force-reinstall
} finally {
    $env:PIP_CACHE_DIR = $PreviousPipCache
}
$Backend = if ($Profile -eq "cuda") { "cuda" } else { "cpu" }
& .venv/Scripts/avril.exe doctor --backend $Backend
Write-Host "Avril is ready. Native Windows CUDA is experimental; WSL2 is recommended."
Write-Host "Avril pip cache: $PipCache"
