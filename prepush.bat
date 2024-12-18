@echo off

:: Define the zip file name
set ZIPFILE=haha.zip

:: Compress the dataset folder
echo Zipping the dataset folder...
powershell Compress-Archive -Path haha -DestinationPath %ZIPFILE% -Force

:: Add the zip file to Git
echo Adding %ZIPFILE% to the repository...
git add %ZIPFILE%
git commit -m "Update dataset.zip before pushing"

:: Push changes to the remote repository
echo Pushing changes to remote...
git push
