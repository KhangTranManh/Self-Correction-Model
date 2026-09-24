param([switch]$SkipPackages)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $projectRoot

$toolsDir = Join-Path $projectRoot '.tools\uv'
$venvDir = Join-Path $projectRoot '.venv'
$env:UV_INSTALL_DIR = $toolsDir
$env:UV_NO_MODIFY_PATH = '1'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.tools\python'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$uvExe = Join-Path $toolsDir 'uv.exe'
$pythonExe = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $uvExe)) {
    Write-Host 'Installing uv from the official Astral installer...'
    Invoke-RestMethod 'https://astral.sh/uv/0.12.17/install.ps1' | Invoke-Expression
}
if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "uv installer did not create $uvExe"
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    & $uvExe venv --python 3.10 $venvDir
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 environment creation failed' }
}

if (-not $SkipPackages) {
    Write-Host 'Installing CUDA 12.8 PyTorch wheel...'
    & $uvExe pip install --python $pythonExe --index-url 'https://download.pytorch.org/whl/cu128' 'torch==2.7.1'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    Write-Host 'Installing pinned Phase 5 inference dependencies...'
    & $uvExe pip install --python $pythonExe 'transformers==4.57.1' 'accelerate==1.10.1' 'peft==0.17.1' 'huggingface_hub==0.35.3' 'python-dotenv==1.1.1' 'sympy==1.14.0' 'safetensors==0.6.2' 'PyYAML==6.0.3'
    if ($LASTEXITCODE -ne 0) { throw 'Phase 5 dependency installation failed' }
}

& $pythonExe 'phase5/scripts/gpu_preflight.py' --output 'phase5/runs/gpu_preflight.json'
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed' }
Write-Host "Project: $projectRoot"
Write-Host "Python: $pythonExe"
Write-Host 'Preflight: phase5\runs\gpu_preflight.json'
