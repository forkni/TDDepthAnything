@echo off
setlocal enabledelayedexpansion

:: Set variables
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "CHECKPOINTS_DIR=%SCRIPT_DIR%checkpoints"
set "PYTHON_PATH=python"
set "INSTALL_FLAG=::INSTALL_COMPLETE_FLAG::"
set "LOG_FILE=%SCRIPT_DIR%installation.log"
set "INSTALL_COMPLETE_FLAG=%SCRIPT_DIR%installation_complete.flag"

:: Start logging
call :log "Installation started"

:: Check for admin privileges
call :check_admin || goto :error

:: Get desired Python version
set /p PYTHON_VERSION=Enter the desired Python version (e.g., 3.11): || set "PYTHON_VERSION=3.11"
call :log "Entered Python version: %PYTHON_VERSION%"

:: Check Python installation
call :check_python || goto :error

:: Get desired CUDA version
set /p CUDA_VERSION=Enter the desired CUDA version (e.g., 118): || set "CUDA_VERSION=118"
call :log "Entered CUDA version: %CUDA_VERSION%"

:: Check CUDA installation
call :check_cuda || goto :error

:: Create and activate virtual environment
call :create_venv || goto :error
call :activate_venv || goto :error

:: Install dependencies
call :install_dependencies || goto :error

:: Create checkpoints directory
call :create_checkpoints_dir || goto :error

:: Set installation complete flag
echo %INSTALL_FLAG% > "%INSTALL_COMPLETE_FLAG%"
call :log "Installation completed successfully"
echo Please place your model in the '%CHECKPOINTS_DIR%' folder before running accelerate.bat.

goto :exit

:check_admin
    net session >nul 2>&1
    if %errorlevel% neq 0 (
        call :log "This script requires administrative privileges"
        exit /b 1
    )
    exit /b 0

:check_python
    where %PYTHON_PATH% >nul 2>&1 || (
        call :log "Python is not found in PATH. Please install Python and add it to PATH"
        exit /b 1
    )
    %PYTHON_PATH% -c "import sys; exit(0 if sys.version_info[:2] == tuple(map(int, '%PYTHON_VERSION%'.split('.'))) else 1)"
    if %errorlevel% neq 0 (
        call :log "Python %PYTHON_VERSION% is not installed or not in PATH"
        exit /b 1
    )
    call :log "Python %PYTHON_VERSION% found"
    exit /b 0

:check_cuda
    where nvcc >nul 2>&1 || (
        call :log "CUDA toolkit not found. Please install CUDA toolkit"
        exit /b 1
    )
    call :log "CUDA toolkit found"
    exit /b 0

:create_venv
    if not exist "%VENV_DIR%" (
        call :log "Creating virtual environment in %VENV_DIR%..."
        %PYTHON_PATH% -m venv "%VENV_DIR%" || (
            call :log "Failed to create virtual environment"
            exit /b 1
        )
    )
    exit /b 0

:activate_venv
    call :log "Activating virtual environment..."
    if not exist "%VENV_DIR%\Scripts\activate.bat" (
        call :log "Virtual environment activation script not found"
        exit /b 1
    )
    call "%VENV_DIR%\Scripts\activate.bat" || (
        call :log "Failed to activate virtual environment"
        exit /b 1
    )
    %PYTHON_PATH% -c "import sys; print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    exit /b 0

:install_dependencies
    call :log "Installing dependencies..."
    call :install_package pip --upgrade
    call :install_package wheel
    call :install_package nvidia-pyindex
    call :install_package nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
    call :install_package tensorrt==9.0.1.post11.dev4 --pre --extra-index-url https://pypi.nvidia.com --no-cache-dir
    call :install_package torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu%CUDA_VERSION%
    if exist "%SCRIPT_DIR%requirements.txt" (
        call :install_package -r "%SCRIPT_DIR%requirements.txt"
    ) else (
        call :log "WARNING: requirements.txt not found. Skipping additional dependencies"
    )
    pip uninstall -y nvidia-cudnn-cu11 || echo WARNING: Failed to uninstall nvidia-cudnn-cu11
    exit /b 0

:install_package
    call :log "Installing %*"
    %PYTHON_PATH% -m pip install %* || (
        call :log "Failed to install %*"
        exit /b 1
    )
    exit /b 0

:create_checkpoints_dir
    if not exist "%CHECKPOINTS_DIR%" (
        call :log "Creating '%CHECKPOINTS_DIR%' folder..."
        mkdir "%CHECKPOINTS_DIR%" || (
            call :log "Failed to create checkpoints directory"
            exit /b 1
        )
    )
    exit /b 0

:log
    echo %~1
    echo %date% %time%: %~1 >> "%LOG_FILE%"
    exit /b 0

:error
    call :log "Error: An error occurred during the installation process"
    call :cleanup
    goto :exit

:cleanup
    call :log "Starting cleanup process..."
    
    if exist "%VENV_DIR%" (
        call :log "Removing virtual environment..."
        rmdir /s /q "%VENV_DIR%"
    )
    
    if exist "%CHECKPOINTS_DIR%" (
        call :log "Removing checkpoints directory..."
        rmdir /s /q "%CHECKPOINTS_DIR%"
    )
    
    if exist "%INSTALL_COMPLETE_FLAG%" (
        call :log "Removing installation complete flag..."
        del "%INSTALL_COMPLETE_FLAG%"
    )
    
    call :log "Cleanup process completed"
    exit /b 0

:exit
    if exist "%INSTALL_COMPLETE_FLAG%" (
        call :log "Installation completed successfully"
    ) else (
        call :log "Installation failed or was incomplete"
    )
    call :log "Script execution finished"
    endlocal
    pause
    exit /b