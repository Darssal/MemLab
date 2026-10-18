Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

try {
  $out = (& docker info --format '{{.ServerVersion}}' 2>&1 | Out-String).Trim()
} catch {
  Stop-FailClosed -Reason "Docker engine is not running."
}
if ($LASTEXITCODE -ne 0 -or -not $out -or $out.ToLowerInvariant().Contains("error") -or $out.ToLowerInvariant().Contains("request returned")) {
  Stop-FailClosed -Reason "Docker engine is not running."
}

Write-Output "Docker engine is running."
