# Weekly data update: refresh every source, rebuild the dataset and the models,
# then commit and push the new data. Run by the scheduled task that
# install_weekly_task.ps1 creates; it can also be run by hand.
#
#   powershell -ExecutionPolicy Bypass -File scripts\weekly_update.ps1

param(
    [string]$Python = "python",   # interpreter with the package installed (pip install -e ".[scrape]")
    [switch]$NoPush               # update and commit, but do not push
)

# Native commands report through their exit codes (checked below); with "Stop",
# Windows PowerShell would abort on any line they write to stderr.
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
New-Item -ItemType Directory -Force "data\processed" | Out-Null
$log = Join-Path $repo "data\processed\weekly_update.log"

function Write-Log($message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $message" | Out-File -FilePath $log -Append -Encoding utf8
}

Write-Log "Update started"
& $Python -m ufc_rating.pipeline --scrape 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Log "Pipeline failed (exit code $LASTEXITCODE); nothing committed"
    exit 1
}

git add dataset data/raw
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Log "No new data"
    exit 0
}
git commit -m "Weekly data update ($(Get-Date -Format 'yyyy-MM-dd'))" 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Log "Commit failed"
    exit 1
}
if (-not $NoPush) {
    git push 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Push failed; the commit is kept locally and will go out with the next push"
        exit 1
    }
}
Write-Log "Update committed"
