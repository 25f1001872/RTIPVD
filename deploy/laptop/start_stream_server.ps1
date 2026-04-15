param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8088,
    [string]$ModelPath = "weights/best.pt",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/laptop_stream_server.py --host $Host --port $Port --model $ModelPath --device $Device
