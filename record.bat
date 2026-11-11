@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" tools\record_test_media.py
pause
