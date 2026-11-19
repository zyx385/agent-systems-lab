param(
    [string]$Output = "artifacts/repomedic-demo-live.json"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$demoRepository = Join-Path $projectRoot "examples/repomedic/demo_repo"
$outputPath = Join-Path $projectRoot $Output

if (-not $env:DASHSCOPE_API_KEY -or -not $env:DASHSCOPE_BASE_URL -or -not $env:DASHSCOPE_MODEL) {
    throw "请先配置DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL和DASHSCOPE_MODEL"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
python -m repomedic.cli $demoRepository `
    --issue "total_price concatenates string prices instead of adding numbers" `
    --traceback 'File "app/pricing.py", line 2, in total_price' `
    --backend dashscope `
    --baseline `
    --json `
    --audit-log $outputPath

exit $LASTEXITCODE
