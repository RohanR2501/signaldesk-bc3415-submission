param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Backend', 'Frontend')]
    [string]$Service,
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$port = if ($Service -eq 'Backend') { 8000 } else { 5173 }
if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
    throw "Port $port is occupied. Use the existing service or stop it in its own terminal; no process was terminated."
}

# Foreground processes keep logs visible and let Ctrl+C stop only this service.
if ($Service -eq 'Backend') {
    $uv = Join-Path $PSScriptRoot '.tools/Scripts/uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) { throw 'Run setup.ps1 first.' }
    if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.runtime/uv-cache' }
    $env:PYTHONUTF8 = '1'
    $extras = @('--extra', 'data', '--extra', 'baselines')
    if (-not $CpuOnly) { $extras += @('--extra', 'gpu') }
    & $uv run --frozen @extras uvicorn signaldesk.main:app --host 127.0.0.1 --port 8000 --no-access-log
} else {
    if (-not (Test-Path -LiteralPath 'frontend/node_modules/.bin/vite.cmd')) {
        throw 'Run setup.ps1 first.'
    }
    & npm.cmd --prefix frontend run dev
}
if ($LASTEXITCODE -ne 0) { throw "$Service exited with code $LASTEXITCODE. See the output above." }
