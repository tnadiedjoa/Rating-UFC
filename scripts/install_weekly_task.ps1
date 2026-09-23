# Create (or replace) the Windows scheduled task that runs weekly_update.ps1
# every Monday at noon. If the computer is off or asleep at that time, the
# task runs as soon as possible afterwards (StartWhenAvailable). It runs in
# the current user's session, so it needs that user to be logged on.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_weekly_task.ps1
#   Unregister-ScheduledTask -TaskName "UFC dataset weekly update"   # to remove it

param(
    [string]$Python = (Get-Command python).Source,
    [string]$Day = "Monday",
    [string]$At = "12:00"
)

$script = Join-Path $PSScriptRoot "weekly_update.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Python `"$Python`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName "UFC dataset weekly update" -Action $action -Trigger $trigger `
    -Settings $settings -Force `
    -Description "Refreshes the UFC dataset (Kaggle, ufcstats.com, Wikipedia, bestfightodds), rebuilds it and pushes it to GitHub." | Out-Null

Write-Output "Scheduled every $Day at $At (runs later if the computer is off): $script"
Write-Output "Python: $Python"
