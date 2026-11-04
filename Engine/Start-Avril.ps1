$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

if (-not (Test-Path ".venv/Scripts/avril.exe")) {
    & "$ProjectDir/setup_avril.ps1" cpu
}

$WebRoot = $env:AVRIL_WEB_ROOT
if (-not $WebRoot) {
    $Candidates = @(
        (Join-Path (Split-Path -Parent $ProjectDir) "Avril-V3.0-Web/dist-pages"),
        (Join-Path $ProjectDir "Avril-V3.0-Web/dist-pages"),
        (Join-Path (Split-Path -Parent $ProjectDir) "Avril-V2.0-Web/dist-pages"),
        (Join-Path $ProjectDir "Avril-V2.0-Web/dist-pages"),
        (Join-Path (Split-Path -Parent $ProjectDir) "Avril-V1.0-Web/dist-pages"),
        (Join-Path $ProjectDir "Avril-V1.0-Web/dist-pages"),
        (Join-Path $ProjectDir "web/dist-pages")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path (Join-Path $Candidate "index.html")) {
            $WebRoot = $Candidate
            break
        }
    }
}

if (-not $WebRoot -or -not (Test-Path (Join-Path $WebRoot "index.html"))) {
    throw "Extract the matching Avril Algorithm and Web packages side by side, then retry."
}

if ($env:AVRIL_DRY_RUN -eq "1") {
    Write-Host ".venv/Scripts/avril.exe serve --web-root `"$WebRoot`" --open-browser"
    exit 0
}

& .venv/Scripts/avril.exe serve --web-root $WebRoot --open-browser
