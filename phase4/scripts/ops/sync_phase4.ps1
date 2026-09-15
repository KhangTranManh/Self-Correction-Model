param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("PushCode", "PullEvidence")]
    [string]$Direction,
    [string]$HostName = "159.48.242.34",
    [int]$Port = 21611,
    [string]$RemoteRoot = "/root/agi",
    [string]$IdentityFile = "$env:USERPROFILE/.ssh/agi_phase4_gpu_ed25519"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Remote = "root@$HostName"

if ($Direction -eq "PushCode") {
    $Archive = Join-Path $env:TEMP "agi_phase4_sync.tar.gz"
    if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive }
    Push-Location $ProjectRoot
    try {
        tar -czf $Archive --exclude='.git' --exclude='.env' --exclude='.env.*' `
            --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' `
            README.md .gitignore instructionAI phase1 phase2 phase3 phase4
    } finally {
        Pop-Location
    }
    scp -i $IdentityFile -P $Port $Archive "${Remote}:/root/agi_phase4_sync.tar.gz"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    ssh -i $IdentityFile -p $Port $Remote `
        "mkdir -p '$RemoteRoot' && tar -xzf /root/agi_phase4_sync.tar.gz -C '$RemoteRoot' && rm /root/agi_phase4_sync.tar.gz"
    Remove-Item -LiteralPath $Archive
    exit $LASTEXITCODE
}

$LocalData = Join-Path $ProjectRoot "phase4/data"
$LocalRuns = Join-Path $ProjectRoot "phase4/runs"
$LocalOutputs = Join-Path $ProjectRoot "outputs"
New-Item -ItemType Directory -Force $LocalData, $LocalRuns, $LocalOutputs | Out-Null
ssh -i $IdentityFile -p $Port $Remote `
    "mkdir -p '$RemoteRoot/phase4/data/rollouts' '$RemoteRoot/phase4/runs' '$RemoteRoot/outputs'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
scp -i $IdentityFile -P $Port -r "${Remote}:${RemoteRoot}/phase4/data/." $LocalData
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
scp -i $IdentityFile -P $Port -r "${Remote}:${RemoteRoot}/phase4/runs/." $LocalRuns
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$RemoteOutputPaths = ssh -i $IdentityFile -p $Port $Remote `
    "find '$RemoteRoot/outputs' -mindepth 1 -maxdepth 1 -type d -name 'phase4_*' -print"
foreach ($RemoteOutputPath in $RemoteOutputPaths) {
    scp -i $IdentityFile -P $Port -r "${Remote}:$RemoteOutputPath" $LocalOutputs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
