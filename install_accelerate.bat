@echo off
setlocal enabledelayedexpansion

:: Get desired Python version
set /p PYTHON_VERSION=Enter the desired Python version (e.g., 3.11): 
echo Entered Python version: %PYTHON_VERSION%

:: Set variables
set VENV_DIR=.venv
set CHECKPOINTS_DIR=checkpoints
set PYTHON_PATH=python

:: Check if Python is installed and in PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not found in PATH. Please install Python and add it to PATH.
    goto :exit
)

:: Check if the specific version of Python is installed
%PYTHON_PATH% -c "import sys; exit(0 if sys.version_info[:2] == tuple(map(int, '%PYTHON_VERSION%'.split('.'))) else 1)"
if %errorlevel% neq 0 (
    echo Python %PYTHON_VERSION% is not installed or not in PATH.
    goto :exit
)

:: Create and activate virtual environment
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    %PYTHON_PATH% -m venv %VENV_DIR% || (
        echo Failed to create virtual environment
        goto :exit
    )
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate || (
    echo Failed to activate virtual environment
    goto :exit
)

:: Verify Python version in virtual environment
python -c "import sys; print(sys.version_info[:2])"

:: Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip || echo "Failed to upgrade pip"
pip install wheel
pip install nvidia-pyindex || echo "Failed to install nvidia-pyindex"
pip install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir || echo "Failed to install nvidia-cudnn-cu11"
pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir || echo "Failed to install tensorrt"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 || echo "Failed to install PyTorch and Torchvision"

if exist requirements.txt (
    pip install -r requirements.txt || echo "Failed to install dependencies from requirements.txt"
) else (
    echo "WARNING: requirements.txt not found. Skipping additional dependencies."
)

pip uninstall -y nvidia-cudnn-cu11 || echo "WARNING: Failed to uninstall nvidia-cudnn-cu11"

:: Create checkpoints directory
if not exist %CHECKPOINTS_DIR% (
    echo Creating 'checkpoints' folder...
    mkdir %CHECKPOINTS_DIR% || echo "Failed to create checkpoints directory"
)

:: Run model acceleration
echo "Preparing for model acceleration..."
python accelerate_model.py || echo "Failed to run accelerate_model.py"

echo "Setup process completed successfully"

:exit
pause
exit /b 0