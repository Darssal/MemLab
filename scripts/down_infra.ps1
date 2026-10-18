Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/_infra_common.ps1"

Invoke-DockerCompose -ComposeArgs @("down")
Write-Output "Infra services stopped."
