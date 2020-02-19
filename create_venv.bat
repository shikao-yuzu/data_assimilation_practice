@echo off
setlocal

set VENV_SCRIPTS=".\venv\Scripts\"

rem create virtual env
python -m venv venv

rem activate virtual env
call %VENV_SCRIPTS%activate.bat

rem install dependency package
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools

pip install numpy==1.18.0
pip install pandas==0.25.3
pip install matplotlib==3.1.2
pip install japanize-matplotlib==1.0.5

rem install this package
python setup.py install

pip freeze

rem deactivate virtual env
call %VENV_SCRIPTS%deactivate.bat

pause

endlocal
