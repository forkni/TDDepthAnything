@echo off
setlocal enabledelayedexpansion
:: Set variables
set VENV_DIR=.venv
set CHECKPOINTS_DIR=checkpoints
set PYTHON_PATH=python
set INSTALL_FLAG=::INSTALL_COMPLETE_FLAG::
:: Get desired Python version
set /p PYTHON_VERSION=Enter the desired Python version (e.g., 3.11):
echo Entered Python version: %PYTHON_VERSION%
:: Check if Python is installed and in PATH
where %PYTHON_PATH% >nul 2>nul
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
%PYTHON_PATH% -m venv %VENV_DIR% || goto :error
)
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate || goto :error
:: Verify Python version in virtual environment
%PYTHON_PATH% -c "import sys; print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
:: Install dependencies
echo Installing dependencies...
call :install_package pip --upgrade
call :install_package wheel
call :install_package nvidia-pyindex
call :install_package nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
call :install_package tensorrt==9.0.1.post11.dev4 --pre --extra-index-url https://pypi.nvidia.com --no-cache-dir
call :install_package torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
if exist requirements.txt (
call :install_package -r requirements.txt
) else (
echo WARNING: requirements.txt not found. Skipping additional dependencies.
)
pip uninstall -y nvidia-cudnn-cu11 || echo WARNING: Failed to uninstall nvidia-cudnn-cu11
:: Create checkpoints directory
if not exist %CHECKPOINTS_DIR% (
echo Creating '%CHECKPOINTS_DIR%' folder...
mkdir %CHECKPOINTS_DIR% || (
echo Failed to create checkpoints directory.
goto :error
)
)
:: Set installation complete flag
echo %INSTALL_FLAG% > installation_complete.flag
echo Installation completed successfully.
echo Please place your model in the '%CHECKPOINTS_DIR%' folder before running accelerate.bat.
goto :exit
:install_package
echo Installing %*
%PYTHON_PATH% -m pip install %* || (
echo Failed to install %*
exit /b 1
)
exit /b 0
:error
echo An error occurred during the installation process.
goto :exit
:exit
pause
exit /b