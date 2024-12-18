@echo off
setlocal enabledelayedexpansion

:: Prompt for the prefix "a"
set /p a=Enter the current prefix (a): 

:: Prompt for the new prefix "b"
set /p b=Enter the new prefix (b): 

:: Loop through the range 0 to 19
for /l %%i in (0,1,19) do (
    set oldName=%a%_%%i.png
    set newName=%b%_%%i.png

    if exist "!oldName!" (
        ren "!oldName!" "!newName!"
        echo Renamed "!oldName!" to "!newName!"
    ) else (
        echo File "!oldName!" not found. Skipping.
    )
)

echo Process completed.
pause
