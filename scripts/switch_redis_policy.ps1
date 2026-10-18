param(
  [ValidateSet("allkeys-lfu", "allkeys-lru")]
  [string]$Policy = "allkeys-lfu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

& "$PSScriptRoot/check_infra.ps1" | Out-Null

$resp = (Invoke-DockerCompose -ComposeArgs @("exec", "-T", "redis", "redis-cli", "-h", "127.0.0.1", "-p", "6379", "CONFIG", "SET", "maxmemory-policy", $Policy) | Out-String).Trim()
if ($resp -notmatch "OK") {
  Stop-FailClosed -Reason "Failed to set Redis maxmemory-policy=$Policy"
}

$resp2 = (Invoke-DockerCompose -ComposeArgs @("exec", "-T", "redis", "redis-cli", "-h", "127.0.0.1", "-p", "6379", "CONFIG", "SET", "maxmemory", "4gb") | Out-String).Trim()
if ($resp2 -notmatch "OK") {
  Stop-FailClosed -Reason "Failed to set Redis maxmemory=4gb."
}

$verify = (Invoke-DockerCompose -ComposeArgs @("exec", "-T", "redis", "redis-cli", "-h", "127.0.0.1", "-p", "6379", "CONFIG", "GET", "maxmemory-policy") | Out-String).Trim()
if ($verify -notmatch $Policy) {
  Stop-FailClosed -Reason "Redis policy verification failed (expected $Policy)."
}

Write-Output "Applied Redis policy: $Policy"
