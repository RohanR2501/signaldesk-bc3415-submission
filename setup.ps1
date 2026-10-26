param(
    [switch]$CpuOnly,
    [switch]$AnalyzerOnly,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$manifest = Get-Content -LiteralPath 'config/runtime-artifacts.json' -Raw | ConvertFrom-Json
function Test-Artifact($Path, $Entry) {
    return (Test-Path -LiteralPath $Path -PathType Leaf) -and
        (Get-Item -LiteralPath $Path).Length -eq $Entry.bytes -and
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Entry.sha256
}

$missingModels = @()
$missingHistory = @()
foreach ($entry in $manifest.files) {
    if ($CpuOnly -and $entry.group -eq 'cuda') { continue }
    if ($AnalyzerOnly -and $entry.group -eq 'historical') { continue }
    $path = Join-Path $PSScriptRoot $entry.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        if ($entry.group -eq 'historical') { $missingHistory += $entry.path }
        else { $missingModels += $entry.path }
    } elseif (-not (Test-Artifact $path $entry)) {
        throw "Checksum/size mismatch: $($entry.path). Restore the frozen file manually; nothing was replaced."
    }
}
if ($VerifyOnly) {
    if ($missingModels.Count -or $missingHistory.Count) {
        throw "Missing runtime files: $(($missingModels + $missingHistory) -join ', '). VerifyOnly does not download files."
    }
    Write-Host 'Required runtime file sizes and SHA256 checksums verified.'
    return
}

foreach ($command in @('py.exe', 'node.exe', 'npm.cmd', 'curl.exe')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Install $command prerequisites listed in README, then retry."
    }
}
& py.exe -3.13 -c "import sys; assert sys.version_info[:2] == (3, 13)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
$nodeVersion = & node.exe --version
if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^v24\.') { throw 'Node.js 24 is required.' }

if ($missingModels.Count) {
    $release = $manifest.model_release
    $cache = Join-Path $PSScriptRoot '.runtime'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $archive = Join-Path $cache 'SignalDesk_Runtime_Models.zip'
    if (-not (Test-Path -LiteralPath $archive)) {
        $partial = "$archive.part"
        if (Test-Path -LiteralPath $partial) { throw "Inspect/remove the incomplete download at $partial, then retry." }
        Write-Host 'Downloading saved models from the public runtime-models-v1 Release (238 MiB)...'
        try {
            & curl.exe --fail --location --retry 2 --connect-timeout 30 --max-time 600 --silent --show-error --output $partial $release.url
            if ($LASTEXITCODE -ne 0) { throw 'Model download failed. Check GitHub/network access and retry setup.' }
            if (-not (Test-Artifact $partial $release)) { throw 'Downloaded model archive failed checksum/size verification.' }
            Move-Item -LiteralPath $partial -Destination $archive
        } finally {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
        }
    }
    if (-not (Test-Artifact $archive $release)) {
        throw "Cached model archive checksum/size mismatch: $archive. Inspect/remove it and retry; nothing was replaced."
    }
    # Extract only a verified archive; preflight every destination before copying missing files.
    $staging = Join-Path $cache ('model-install-' + [guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $archive -DestinationPath $staging
    $bundle = Get-Content -LiteralPath (Join-Path $staging 'runtime-model-checksums.json') -Raw | ConvertFrom-Json
    foreach ($entry in $bundle.files) {
        $source = Join-Path $staging $entry.path
        $destination = Join-Path $PSScriptRoot $entry.path
        if (-not (Test-Artifact $source $entry)) { throw "Invalid bundled file: $($entry.path)" }
        if ((Test-Path -LiteralPath $destination) -and -not (Test-Artifact $destination $entry)) {
            throw "Existing file differs: $($entry.path). Nothing was replaced; staging retained at $staging."
        }
    }
    foreach ($entry in $bundle.files) {
        $destination = Join-Path $PSScriptRoot $entry.path
        if (-not (Test-Path -LiteralPath $destination)) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            [System.IO.File]::Copy((Join-Path $staging $entry.path), $destination, $false)
        }
    }
    $resolvedStaging = (Resolve-Path -LiteralPath $staging).Path
    if (-not $resolvedStaging.StartsWith($cache + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected staging location; inspect $resolvedStaging manually."
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse
}
foreach ($entry in $manifest.files) {
    if ($entry.group -eq 'historical' -or ($CpuOnly -and $entry.group -eq 'cuda')) { continue }
    if (-not (Test-Artifact (Join-Path $PSScriptRoot $entry.path) $entry)) {
        throw "Model verification failed: $($entry.path)"
    }
}
Write-Host 'Required saved model sizes and SHA256 checksums verified.'
$partialSetup = $AnalyzerOnly -or $missingHistory.Count -gt 0
if ($partialSetup) {
    Write-Warning "PARTIALLY REPRODUCIBLE: Analyzer available after dependency installation; historical views need four separately authorised files. Missing: $($missingHistory -join ', ')"
}

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
if ($LASTEXITCODE -ne 0) { throw "Locked frontend dependency installation failed. Stop this checkout's frontend with Ctrl+C before retrying; inspect npm's error above." }
if ($partialSetup) { Write-Warning 'Setup complete for Analyzer only. The populated historical application is NOT ready; see README.' }
else { Write-Host 'All frozen runtime files verified; complete application dependencies installed.' }
Write-Host 'Run start.ps1 Backend and start.ps1 Frontend in separate terminals.'
