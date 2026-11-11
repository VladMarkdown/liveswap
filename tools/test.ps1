# 30 second speed test with the webcam. Asks for one photo, shows the numbers
# live and saves everything to "liveswap-test-log.txt" on the desktop.
# ASCII-only on purpose (Windows PowerShell 5.1).

$ErrorActionPreference = 'Continue'   # Python logs to stderr; that is not a failure
$Dir = Split-Path -Parent $PSScriptRoot
$py  = Join-Path $Dir '.venv\Scripts\python.exe'
$log = Join-Path ([Environment]::GetFolderPath('Desktop')) 'liveswap-test-log.txt'

Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose a face photo (of a person who has agreed to this)'
$dialog.Filter = 'Images|*.jpg;*.jpeg;*.png;*.bmp;*.webp'
if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit }

Set-Location $Dir
$env:PYTHONIOENCODING = 'utf-8'
Write-Host 'Running for 30 seconds. Look at the camera.' -ForegroundColor Cyan
& $py 'tools\check_install.py' 2>&1 | Tee-Object -FilePath $log
& $py 'tools\bench.py' '--photo' $dialog.FileName '--source' '0' '--seconds' '30' 2>&1 |
    Tee-Object -FilePath $log -Append

Write-Host ''
Write-Host "Finished. Please send this file to the developer: $log" -ForegroundColor Green
explorer.exe "/select,$log"
Read-Host 'Press Enter to close'
