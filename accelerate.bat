@echo off
setlocal enabledelayedexpansion

:: Set variables
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "CHECKPOINTS_DIR=%SCRIPT_DIR%checkpoints"
set "INSTALL_FLAG=::INSTALL_COMPLETE_FLAG::"
set "ACCELERATION_SCRIPT=accelerate_model.py"
set "LOG_FILE=%SCRIPT_DIR%acceleration.log"

:: Parse command-line arguments
if "%~1" neq "" set "ACCELERATION_SCRIPT=%~1"
if "%~2" neq "" set "MODEL_FILE=%~2"
if "%~3" neq "" set "OUTPUT_DIR=%~3"

:: Start logging
call :log "Acceleration process started"

:: Check if installation was completed
if not exist "%SCRIPT_DIR%installation_complete.flag" (
    call :log "Installation has not been completed. Please run install.bat first."
    goto :error
)

:: Activate virtual environment
call :log "Activating virtual environment..."
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    call :log "Virtual environment not found. Please run install.bat again."
    goto :error
)
call "%VENV_DIR%\Scripts\activate.bat" || (
    call :log "Failed to activate virtual environment. Please run install.bat again."
    goto :error
)

:: Check Python version and CUDA availability
python -c "import sys; assert sys.version_info >= (3, 7), 'Python 3.7 or higher is required'" || (
    call :log "Python 3.7 or higher is required."
    goto :error
)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'" || (
    call :log "CUDA is not available. Please check your CUDA installation."
    goto :error
)

:: Check if checkpoints directory exists and is not empty
if not exist "%CHECKPOINTS_DIR%" (
    call :log "Checkpoints directory '%CHECKPOINTS_DIR%' not found. Please run install.bat first."
    goto :error
)
if not exist "%CHECKPOINTS_DIR%\*" (
    call :log "Checkpoints directory '%CHECKPOINTS_DIR%' is empty. Please place your model in this directory before running acceleration."
    goto :error
)

:: Check if acceleration script exists
if not exist "%ACCELERATION_SCRIPT%" (
    call :log "Acceleration script '%ACCELERATION_SCRIPT%' not found."
    goto :error
)

:: Run model acceleration
call :log "Preparing for model acceleration..."
call :log "Starting acceleration process. This may take a while..."

set "COMMAND=python "%ACCELERATION_SCRIPT%""
if defined MODEL_FILE set "COMMAND=%COMMAND% --model "%MODEL_FILE%""
if defined OUTPUT_DIR set "COMMAND=%COMMAND% --output "%OUTPUT_DIR%""

%COMMAND%

if %errorlevel% neq 0 (
    call :log "Failed to run %ACCELERATION_SCRIPT%. Please check the output above for details."
    goto :error
)

call :log "Model acceleration completed successfully."
goto :exit

:log
echo %~1
echo %date% %time%: %~1 >> "%LOG_FILE%"
exit /b 0

:error
call :log "An error occurred during the acceleration process."
goto :exit

:exit
call :log "Acceleration process finished."
endlocal
pause
exit /b