param(
  [ValidateSet("B0", "B1", "V1", "F1", "F2", "A1", "A2")]
  [string]$Variant = "B1",
  [int]$Sessions = 200,
  [int]$Repeat = 0,
  [int]$Seed = 42,
  [int]$WorkerProcesses = 1,
  [int]$SessionsPerProcess = 200,
  [int]$OpsPerSession = 50,
  [string]$TracePath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

& "$PSScriptRoot/check_docker.ps1"
& "$PSScriptRoot/check_infra.ps1"

if ($Variant -eq "A2") {
  & "$PSScriptRoot/switch_redis_policy.ps1" -Policy "allkeys-lru"
} else {
  & "$PSScriptRoot/switch_redis_policy.ps1" -Policy "allkeys-lfu"
}

$cmd = @(
  "-m", "memlab.runner_stub",
  "--variant", $Variant,
  "--n_sessions", "$Sessions",
  "--repeat_idx", "$Repeat",
  "--seed", "$Seed",
  "--worker_processes", "$WorkerProcesses",
  "--sessions_per_process", "$SessionsPerProcess",
  "--ops_per_session", "$OpsPerSession",
  "--workload_id", "SYN-200"
)
if ($TracePath) {
  $cmd += @("--trace_path", $TracePath)
}

python @cmd
if ($LASTEXITCODE -ne 0) {
  Stop-FailClosed -Reason "Stub runner failed."
}
