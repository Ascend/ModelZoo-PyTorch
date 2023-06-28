@echo off
for /f "tokens=2 delims==" %%a in ('type ..\..\..\url.ini ^| findstr "zip_url="') do set url=%%a
curl -k %url% -O
if errorlevel 1 exit /b 1

start /wait 7z1805-x64.exe /S
if errorlevel 1 exit /b 1

set "PATH=%ProgramFiles%\7-Zip;%PATH%"
