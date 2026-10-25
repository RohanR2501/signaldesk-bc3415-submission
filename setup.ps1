param(
    [switch]$CpuOnly,
    [switch]$AnalyzerOnly,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Check the frozen files before downloading dependencies; never repair or replace them.
$manifest = Get-Content -LiteralPath 'config/runtime-artifacts.json' -Raw | ConvertFrom-Json
$problems = @()
foreach ($entry in $manifest.files) {
    if ($CpuOnly -and $entry.group -eq 'cuda') { continue }
    if ($AnalyzerOnly -and $entry.group -eq 'historical') { continue }
    $path = Join-Path $PSScriptRoot $entry.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $problems += "Missing: $($entry.path)"
    } elseif ((Get-Item -LiteralPath $path).Length -ne $entry.bytes -or
              (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.sha256) {
        $problems += "Checksum/size mismatch: $($entry.path)"
    }
}
if ($problems.Count) {
    $problems | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    throw 'Runtime verification failed. See README Quick Start; no files were replaced.'
}
if ($AnalyzerOnly) {
    Write-Warning 'Analyzer-only setup: historical views require separately authorised historical files.'
}
Write-Host 'Required runtime file sizes and SHA256 checksums verified.'
if ($VerifyOnly) { return }

foreach ($command in @('py.exe', 'node.exe', 'npm.cmd')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Install $command prerequisites listed in README, then retry."
    }
}
& py.exe -3.13 -c "import sys; assert sys.version_info[:2] == (3, 13)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
$nodeVersion = & node.exe --version
if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^v24\.') { throw 'Node.js 24 is required.' }

$uv = Join-Path $PSScriptRoot '.tools/Scripts/uv.exe'
if (-not (Test-Path -LiteralPath $uv)) {
    if (Test-Path -LiteralPath '.tools') {
        throw '.tools exists without uv.exe. Inspect it before installing uv 0.13.5; nothing was overwritten.'
    }
    & py.exe -3.13 -m venv .tools
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated uv tool environment.' }
    & '.tools/Scripts/python.exe' -m pip install uv==0.13.5
    if ($LASTEXITCODE -ne 0) { throw 'Could not install uv 0.13.5.' }
}
$uvVersion = & $uv --version
if ($LASTEXITCODE -ne 0 -or $uvVersion -notmatch '^uv 0\.13\.5(?:\s|$)') {
    throw 'Use uv 0.13.5 in .tools; the existing installation was not changed.'
}

if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.runtime/uv-cache' }
$env:PYTHONUTF8 = '1'
$extras = @('--extra', 'data', '--extra', 'baselines')
if (-not $CpuOnly) { $extras += @('--extra', 'gpu') }
& $uv sync --frozen --python 3.13 --all-groups @extras
if ($LASTEXITCODE -ne 0) { throw 'Locked Python dependency installation failed.' }
& npm.cmd --prefix frontend ci
if ($LASTEXITCODE -ne 0) { throw 'Locked frontend dependency installation failed.' }
Write-Host 'Setup complete. Run start.ps1 Backend and start.ps1 Frontend in separate terminals.'
