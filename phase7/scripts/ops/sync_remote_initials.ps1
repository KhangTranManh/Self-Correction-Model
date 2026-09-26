param(
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$key = Join-Path $project 'outputs/phase7_transfer/sync_key'
$target = Join-Path $project 'outputs/phase7_remote_v100'
New-Item -ItemType Directory -Force -Path $target | Out-Null
$remote = 'root@159.48.242.1:/root/AGI_phase7/outputs'

while ($true) {
    foreach ($item in @(
        @('phase7_vllm_smoke/report.json', 'smoke_report.json'),
        @('phase7_vllm_smoke/run.log', 'smoke_run.log'),
        @('phase7_initials_v1/initial_rollouts.audit.jsonl', 'initial_rollouts.audit.jsonl'),
        @('phase7_initials_v1/initial_rollouts.jsonl', 'initial_rollouts.jsonl'),
        @('phase7_initials_v1/summary.json', 'initial_summary.json'),
        @('phase7_initials_runner.log', 'initial_runner.log')
    )) {
        $temporary = Join-Path $target ($item[1] + '.part')
        $destination = Join-Path $target $item[1]
        $ErrorActionPreference = 'Continue'
        & scp -q -i $key -o BatchMode=yes -P 25018 "$remote/$($item[0])" $temporary 2>$null
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -eq 0) {
            Move-Item -LiteralPath $temporary -Destination $destination -Force
        }
    }
    if (Test-Path (Join-Path $target 'initial_summary.json')) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
