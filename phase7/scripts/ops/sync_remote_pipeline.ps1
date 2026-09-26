param(
    [int]$IntervalSeconds = 180,
    [string]$RemoteHost = '159.48.242.1',
    [int]$RemotePort = 25018,
    [string]$LocalName = 'phase7_remote_v100',
    [string]$PipelineLog = 'phase7_pipeline.log'
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$key = Join-Path $project 'outputs/phase7_transfer/sync_key'
$remote = "root@${RemoteHost}:/root/AGI_phase7"
$localRoot = Join-Path (Join-Path $project 'outputs') $LocalName
New-Item -ItemType Directory -Force -Path $localRoot | Out-Null

function Copy-RemoteFile([string]$relative) {
    $destination = Join-Path $localRoot ($relative -replace '/', '\')
    $folder = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $temporary = $destination + '.part'
    $ErrorActionPreference = 'Continue'
    & scp -q -i $key -o BatchMode=yes -P $RemotePort "$remote/$relative" $temporary 2>$null
    $status = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($status -eq 0) {
        Move-Item -LiteralPath $temporary -Destination $destination -Force
    }
}

while ($true) {
    Copy-RemoteFile "outputs/$PipelineLog"
    foreach ($name in @('split_report.json', 'development_ids.json', 'protected_ids.json')) {
        Copy-RemoteFile "phase7/data/split_v1/$name"
    }
    Copy-RemoteFile 'phase7/data/protocol/paired_generation_v1_lock.json'
    foreach ($split in @('development', 'protected')) {
        Copy-RemoteFile "outputs/phase7_paired_v1/$split/analysis.json"
        foreach ($checkpoint in @('original_solver', 'warmstart_v2', 'correction_sft_v3')) {
            $prefix = "outputs/phase7_paired_v1/$split/$checkpoint"
            foreach ($name in @('paired_outputs.audit.jsonl', 'paired_outputs.jsonl', 'summary.json')) {
                Copy-RemoteFile "$prefix/$name"
            }
        }
    }
    if (Test-Path (Join-Path $localRoot 'outputs/phase7_paired_v1/protected/analysis.json')) {
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
}
