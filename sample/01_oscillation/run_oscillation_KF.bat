@echo off
setlocal

set VENV_SCRIPTS="..\..\venv\Scripts\"

rem activate virtual env
call %VENV_SCRIPTS%activate.bat

rem run
python oscillation_KF.py

rem deactivate virtual env
call %VENV_SCRIPTS%deactivate.bat

endlocal
pause
