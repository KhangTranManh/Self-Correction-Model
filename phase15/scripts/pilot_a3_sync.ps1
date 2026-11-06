$ErrorActionPreference = 'Continue'
$sessionName = 'phase15-matched'
$remoteRoot = '/content/p15a3/run_a3'
$localRoot = 'C:\Users\khang\Documents\Phase15-recovered-2026-11-03\P15-A3'
$deadlineUtc = [DateTime]::UtcNow.AddHours(7)
New-Item -ItemType Directory -Path $localRoot -Force | Out-Null

while ([DateTime]::UtcNow -lt $deadlineUtc) {
    & colab download -s $sessionName "$remoteRoot/audit.jsonl" "$localRoot/audit_live.jsonl" *> $null
    $listing = & colab ls -s $sessionName $remoteRoot 2>$null
    if ($listing -match 'summary.json') {
        & colab download -s $sessionName "$remoteRoot/summary.json" "$localRoot/summary.json" *> $null
        & colab download -s $sessionName "$remoteRoot/audit.jsonl" "$localRoot/audit.jsonl" *> $null
        break
    }
    if ($listing -match 'failure.json') {
        & colab download -s $sessionName "$remoteRoot/failure.json" "$localRoot/failure.json" *> $null
        break
    }
    Start-Sleep -Seconds 120
}
