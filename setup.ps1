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

function Install-RuntimeBundle($Release, $Entries, [switch]$ModelBundle) {
    $cache = Join-Path $PSScriptRoot '.runtime'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $name = [IO.Path]::GetFileName(([uri]$Release.url).AbsolutePath)
    $archive = Join-Path $cache $name
    if (-not (Test-Path -LiteralPath $archive)) {
        $partial = "$archive.part"
        if (Test-Path -LiteralPath $partial) { throw "Inspect/remove the incomplete download at $partial, then retry." }
        Write-Host "Downloading $name from the public GitHub Release..."
        try {
            & curl.exe --fail --location --retry 2 --connect-timeout 30 --max-time 600 --silent --show-error --output $partial $Release.url
            if ($LASTEXITCODE -ne 0) { throw "Download failed: $name. Check GitHub/network access and retry." }
            if (-not (Test-Artifact $partial $Release)) { throw "Archive checksum/size mismatch: $name" }
            Move-Item -LiteralPath $partial -Destination $archive
        } finally {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
        }
    }
    if (-not (Test-Artifact $archive $Release)) {
        throw "Cached archive checksum/size mismatch: $archive. Inspect/remove it and retry; nothing was replaced."
    }
    # Verify the archive before extraction, and all destinations before installing missing files.
    $staging = Join-Path $cache ('bundle-install-' + [guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $archive -DestinationPath $staging
    if ($ModelBundle) {
        # Retain upstream licence notices alongside the saved model files.
        $Entries = (Get-Content -LiteralPath (Join-Path $staging 'runtime-model-checksums.json') -Raw | ConvertFrom-Json).files
    }
    foreach ($entry in $Entries) {
        $source = Join-Path $staging $entry.path
        $destination = Join-Path $PSScriptRoot $entry.path
        if (-not (Test-Artifact $source $entry)) { throw "Invalid bundled file: $($entry.path)" }
        if ((Test-Path -LiteralPath $destination) -and -not (Test-Artifact $destination $entry)) {
            throw "Existing file differs: $($entry.path). Nothing was replaced; staging retained at $staging."
        }
    }
    foreach ($entry in $Entries) {
        $destination = Join-Path $PSScriptRoot $entry.path
        if (-not (Test-Path -LiteralPath $destination)) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            [IO.File]::Copy((Join-Path $staging $entry.path), $destination, $false)
        }
    }
    $resolvedStaging = (Resolve-Path -LiteralPath $staging).Path
    if (-not $resolvedStaging.StartsWith($cache + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected staging location; inspect $resolvedStaging manually."
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse
}

$required = @($manifest.files | Where-Object {
    (-not $CpuOnly -or $_.group -ne 'cuda') -and
    (-not $AnalyzerOnly -or $_.group -ne 'historical')
})
$missing = @()
foreach ($entry in $required) {
    $path = Join-Path $PSScriptRoot $entry.path
    if (-not (Test-Path -LiteralPath $path)) { $missing += $entry }
    elseif (-not (Test-Artifact $path $entry)) {
        throw "Checksum/size mismatch: $($entry.path). Restore the frozen file manually; nothing was replaced."
    }
}
if ($VerifyOnly) {
    if ($missing.Count) { throw "Missing runtime files: $(($missing.path) -join ', '). VerifyOnly does not download files." }
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

if (@($missing | Where-Object { $_.group -ne 'historical' }).Count) {
    Install-RuntimeBundle $manifest.model_release @($required | Where-Object { $_.group -ne 'historical' }) -ModelBundle
}
if (@($missing | Where-Object { $_.group -eq 'historical' }).Count) {
    Install-RuntimeBundle $manifest.historical_release @($required | Where-Object { $_.group -eq 'historical' })
}
foreach ($entry in $required) {
    if (-not (Test-Artifact (Join-Path $PSScriptRoot $entry.path) $entry)) {
        throw "Runtime verification failed: $($entry.path)"
    }
}
Write-Host 'Required saved models and historical files verified.'

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
if ($AnalyzerOnly) { Write-Host 'Analyzer setup complete; historical files were not required in this mode.' }
else { Write-Host 'Complete application setup finished: saved models and historical data verified.' }
Write-Host 'Run start.ps1 Backend and start.ps1 Frontend in separate PowerShell windows.'
