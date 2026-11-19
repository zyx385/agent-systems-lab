$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvConfig = Join-Path $projectRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $venvConfig)) {
    throw "未找到项目虚拟环境：$venvConfig"
}
$homeLine = Get-Content $venvConfig | Where-Object { $_ -like "home = *" } | Select-Object -First 1
$pythonHome = $homeLine.Substring("home = ".Length).Trim()
$python = Join-Path $pythonHome "python.exe"
if (-not (Test-Path $python)) {
    throw "未找到Python：$python"
}

$env:PYTHONPATH = "$projectRoot;$projectRoot\.venv\Lib\site-packages"
Push-Location $projectRoot
try {
    & $python -m redisdoctor.cli `
        "How does the Redis 7 multi-part AOF manifest work?" `
        --corpus benchmarks\redisdoctor\corpus.json `
        --version 7.2 `
        --minimum-score 4.0
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
