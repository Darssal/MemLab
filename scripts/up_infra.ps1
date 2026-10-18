Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

& "$PSScriptRoot/check_docker.ps1"

Write-Output "Starting infra services (Redis/Postgres/Qdrant)..."
Invoke-DockerCompose -ComposeArgs @("up", "-d")

$deadline = (Get-Date).AddMinutes(5)
$required = Get-RequiredServices
while ((Get-Date) -lt $deadline) {
  $allHealthy = $true
  foreach ($svc in $required) {
    $cid = (Invoke-DockerCompose -ComposeArgs @("ps", "-q", $svc) | Out-String).Trim()
    if (-not $cid) {
      $allHealthy = $false
      break
    }
    $health = (& docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' $cid 2>$null | Out-String).Trim().ToLowerInvariant()
    if ($health -ne "healthy" -and $health -ne "none") {
      $allHealthy = $false
      break
    }
  }
  if ($allHealthy) {
    Write-Output "Infra services report healthy state."
    & "$PSScriptRoot/check_infra.ps1"
    exit 0
  }
  Start-Sleep -Seconds 2
}

Stop-FailClosed -Reason "Infra services did not become healthy within timeout."
