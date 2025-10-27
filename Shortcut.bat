@echo off
REM Create Desktop Shortcut for GridWorld Q-Learning (Windows)

echo ============================================
echo Creating Desktop Shortcut
echo ============================================
echo.

REM Get current directory
set CURRENT_DIR=%CD%

REM Get Desktop path
set DESKTOP=%USERPROFILE%\Desktop

REM Create VBS script to create shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%DESKTOP%\GridWorld Q-Learning.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%CURRENT_DIR%\quickstart.bat" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%CURRENT_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "GridWorld Q-Learning Training Platform" >> CreateShortcut.vbs
echo oLink.IconLocation = "C:\Windows\System32\SHELL32.dll,165" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

REM Execute VBS script
cscript //nologo CreateShortcut.vbs

REM Clean up
del CreateShortcut.vbs

echo.
echo ✅ Desktop shortcut created successfully!
echo.
echo You can now double-click "GridWorld Q-Learning" on your desktop to launch the app.
echo.
pause