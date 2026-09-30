# Creates a Python venv with CUDA PyTorch + Ultralytics.  Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Venv .venv
#
# The default puts the venv on the internal SSD: this project lives on a USB hard drive, and
# importing PyTorch (~5 GB of CUDA DLLs) from it takes minutes - once per dataloader worker, too.
param([string]$Venv = "$env:USERPROFILE\.venvs\avx-dragonfruit")
$ErrorActionPreference = "Stop"

$py = (py -3.12 -c "import sys; print(sys.executable)")   # the Python version this project was verified with
if (-not (Test-Path $Venv)) { & $py -m venv $Venv }
$vpy = Join-Path $Venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8: needed for RTX 50xx
& $vpy -m pip install ultralytics==8.4.166
& $vpy -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('ultralytics', ultralytics.__version__)"
Write-Host "`nActivate with:  & '$Venv\Scripts\Activate.ps1'"
