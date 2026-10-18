param(
  [int]$Requests = 200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

& "$PSScriptRoot/check_docker.ps1"
& "$PSScriptRoot/check_infra.ps1"

python -m memlab.microbench --requests $Requests --out logs/microbench.json
if ($LASTEXITCODE -ne 0) {
  Stop-FailClosed -Reason "Microbench failed."
}
Write-Output "Microbench complete: logs/microbench.json"
