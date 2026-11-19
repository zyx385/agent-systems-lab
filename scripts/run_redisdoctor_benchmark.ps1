$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$homeLine = Get-Content (Join-Path $projectRoot ".venv\pyvenv.cfg") |
    Where-Object { $_ -like "home = *" } |
    Select-Object -First 1
$python = Join-Path $homeLine.Substring("home = ".Length).Trim() "python.exe"
$env:PYTHONPATH = "$projectRoot;$projectRoot\.venv\Lib\site-packages"

Push-Location $projectRoot
try {
    & $python benchmarks\redisdoctor\run_benchmark.py `
        --output artifacts\redisdoctor-benchmark.json `
        --summary artifacts\redisdoctor-benchmark.md
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
