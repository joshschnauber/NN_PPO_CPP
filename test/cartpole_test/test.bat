@echo off
:loop
cp.exe 10000
if %errorlevel% neq 0 (
    echo EXIT FAIL
    echo:
    exit /b %errorlevel%
)
echo EXIT SUCCEED
echo:
goto loop