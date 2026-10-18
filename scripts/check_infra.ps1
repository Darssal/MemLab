Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

& "$PSScriptRoot/check_docker.ps1" | Out-Null

$required = Get-RequiredServices
$composeServiceList = Invoke-DockerCompose -ComposeArgs @("ps", "--services", "--filter", "status=running")
$running = @{}
foreach ($svc in $composeServiceList) {
  $name = ($svc | Out-String).Trim()
  if ($name) {
    $running[$name] = $true
  }
}

foreach ($svc in $required) {
  if (-not $running.ContainsKey($svc)) {
    Stop-FailClosed -Reason "Container '$svc' is not running."
  }
  $cid = (Invoke-DockerCompose -ComposeArgs @("ps", "-q", $svc) | Out-String).Trim()
  if (-not $cid) {
    Stop-FailClosed -Reason "Unable to resolve container id for '$svc'."
  }
  $health = (& docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' $cid 2>$null | Out-String).Trim().ToLowerInvariant()
  if ($health -ne "healthy" -and $health -ne "none") {
    Stop-FailClosed -Reason "Container '$svc' is not healthy."
  }
}

$ports = @{
  redis = 6379
  postgres = 5432
  qdrant = 6333
}
foreach ($svc in $required) {
  $p = $ports[$svc]
  if (-not (Test-TcpPort -Port $p)) {
    Stop-FailClosed -Reason "Service '$svc' is unreachable on 127.0.0.1:$p."
  }
}

$redisVersionRaw = Invoke-DockerCompose -ComposeArgs @("exec", "-T", "redis", "redis-cli", "-h", "127.0.0.1", "-p", "6379", "INFO", "server")
$redisVersion = (($redisVersionRaw | Select-String "redis_version:").Line -replace "redis_version:", "").Trim()
if (-not $redisVersion) {
  Stop-FailClosed -Reason "Unable to read Redis version from Docker container."
}

$pgVersion = (Invoke-DockerCompose -ComposeArgs @("exec", "-T", "postgres", "sh", "-c", 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SHOW server_version;"') | Out-String).Trim()
if (-not $pgVersion) {
  Stop-FailClosed -Reason "Unable to read Postgres version from Docker container."
}

try {
  $qdrantRoot = Invoke-RestMethod -Uri "http://127.0.0.1:6333/" -Method Get -TimeoutSec 5
  $qdrantVersion = [string]$qdrantRoot.version
} catch {
  Stop-FailClosed -Reason "Unable to read Qdrant version from Docker container."
}
if (-not $qdrantVersion) {
  Stop-FailClosed -Reason "Qdrant did not return a version."
}

$probe = (Invoke-DockerCompose -ComposeArgs @("exec", "-T", "redis", "redis-cli", "-h", "127.0.0.1", "-p", "6379", "CONFIG", "SET", "maxmemory-policy", "allkeys-lfu") | Out-String).Trim()
if ($probe -notmatch "OK") {
  Stop-FailClosed -Reason "Redis allkeys-lfu probe failed; legacy alias A2 (F1+LRU ablation) would be invalid."
}

$digests = @{}
foreach ($svc in $required) {
  $cid = (Invoke-DockerCompose -ComposeArgs @("ps", "-q", $svc) | Out-String).Trim()
  if (-not $cid) {
    Stop-FailClosed -Reason "Unable to resolve container id for '$svc'."
  }
  $imageRef = (& docker inspect --format '{{.Config.Image}}' $cid 2>$null | Out-String).Trim()
  if (-not $imageRef) {
    Stop-FailClosed -Reason "Unable to resolve image ref for '$svc'."
  }
  $digest = (& docker image inspect $imageRef --format '{{index .RepoDigests 0}}' 2>$null | Out-String).Trim()
  if (-not $digest) {
    $digest = $imageRef
  }
  $digests[$svc] = $digest
}

Write-Output ("Redis version: {0}" -f $redisVersion)
Write-Output ("Postgres version: {0}" -f $pgVersion)
Write-Output ("Qdrant version: {0}" -f $qdrantVersion)
Write-Output ("Redis image digest: {0}" -f $digests["redis"])
Write-Output ("Postgres image digest: {0}" -f $digests["postgres"])
Write-Output ("Qdrant image digest: {0}" -f $digests["qdrant"])
Write-Output "Infra check passed."
