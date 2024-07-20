@echo off
setlocal enabledelayedexpansion
:: Set variables
set VENV_DIR=.venv
set CHECKPOINTS_DIR=checkpoints
set INSTALL_FLAG=::INSTALL_COMPLETE_FLAG::
set ACCELERATION_SCRIPT=accelerate_model.py
:: Parse command-line arguments
if "%~1" neq "" set ACCELERATION_SCRIPT=%~1
:: Check if installation was completed
if not exist installation_complete.flag (
echo Installation has not been completed. Please run install.bat first.
goto :exit
)
:: Activate virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate || (
echo Failed to activate virtual environment. Please run install.bat again.
goto :exit
)
:: Check if checkpoints directory exists and is not empty
if not exist %CHECKPOINTS_DIR% (
echo Checkpoints directory '%CHECKPOINTS_DIR%' not found. Please run install.bat first.
goto :exit
)
if not exist "%CHECKPOINTS_DIR%*" (
echo Checkpoints directory '%CHECKPOINTS_DIR%' is empty. Please place your model in this directory before running acceleration.
goto :exit
)
:: Check if acceleration script exists
if not exist %ACCELERATION_SCRIPT% (
echo Acceleration script '%ACCELERATION_SCRIPT%' not found.
goto :exit
)
:: Run model acceleration
echo Preparing for model acceleration...
echo Starting acceleration process. This may take a while...
python %ACCELERATION_SCRIPT%
if %errorlevel% neq 0 (
echo Failed to run %ACCELERATION_SCRIPT%. Please check the output above for details.
goto :exit
)
echo Model acceleration completed successfully.
:exit
pause
exit /b