Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ComposeArgs {
  return @("-f", "infra/docker-compose.yml")
}

function Get-RequiredServices {
  return @("redis", "postgres", "qdrant")
}

function Stop-FailClosed {
  param(
    [string]$Reason
  )
  Write-Error "STOP: required services missing; refusing fallback."
  if ($Reason) {
    Write-Error $Reason
  }
  exit 1
}

function Invoke-DockerCompose {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$ComposeArgs
  )
  $all = (Get-ComposeArgs) + $ComposeArgs
  & docker compose @all
}

function Test-TcpPort {
  param(
    [Parameter(Mandatory = $true)]
    [int]$Port
  )
  $r = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
  return [bool]$r.TcpTestSucceeded
}
