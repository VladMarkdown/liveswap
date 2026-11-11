# liveswap installer for a clean Windows PC (nothing preinstalled).
#
# Run from PowerShell (or Win+R):
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/VladMarkdown/liveswap/main/install.ps1 | iex"
#
# It needs no administrator rights and installs everything into one folder
# (C:\liveswap): the app, its own Python, the Python packages (including the
# NVIDIA CUDA/cuDNN runtime) and the models. Running it again updates the app
# and keeps the models. To remove everything, delete that folder and the two
# desktop shortcuts.
#
# Optional environment variables: LIVESWAP_BRANCH (default main),
# LIVESWAP_DIR (default C:\liveswap). Keep the folder path ASCII-only.
#
# This file is ASCII-only on purpose: Windows PowerShell 5.1 can garble
# other characters in a script fetched over HTTP.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes downloads many times slower
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo   = 'VladMarkdown/liveswap'
$Branch = if ($env:LIVESWAP_BRANCH) { $env:LIVESWAP_BRANCH } else { 'main' }
$Dir    = if ($env:LIVESWAP_DIR) { $env:LIVESWAP_DIR } else { Join-Path $env:SystemDrive 'liveswap' }
$Log    = Join-Path $Dir 'install-log.txt'

function Step($text) {
    Write-Host ''
    Write-Host "==> $text" -ForegroundColor Cyan
}

# Run a program and stop the installer if it fails.
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $exe $($arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force -Path $Dir | Out-Null
try { Start-Transcript -Path $Log -Force | Out-Null } catch { }

try {
    Step "Installing liveswap into $Dir (about 6 GB of downloads, 10-30 minutes)"

    Step 'Checking the graphics card'
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) {
        & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    } else {
        Write-Host 'No NVIDIA driver found. The app will install but run slowly on the CPU.' -ForegroundColor Yellow
    }

    Step 'Downloading the application'
    $tmp = Join-Path $env:TEMP ("liveswap-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $zip = Join-Path $tmp 'source.zip'
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $source = Get-ChildItem -Path $tmp -Directory | Select-Object -First 1
    Copy-Item -Path (Join-Path $source.FullName '*') -Destination $Dir -Recurse -Force
    Remove-Item -Recurse -Force $tmp

    Step 'Installing the uv package manager'
    # Everything uv needs stays inside the install folder, on an ASCII path.
    $env:UV_INSTALL_DIR        = Join-Path $Dir '.uv'
    $env:UV_NO_MODIFY_PATH     = '1'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Dir '.python'
    $env:UV_CACHE_DIR          = Join-Path $Dir '.uv-cache'
    $uv = Get-ChildItem -Path $env:UV_INSTALL_DIR -Recurse -Filter 'uv.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $uv) {
        # In a child process: uv's installer requires a permissive execution
        # policy and calls `exit` on failure, which would close this window.
        Run 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
                           'irm https://astral.sh/uv/install.ps1 | iex')
        $uv = Get-ChildItem -Path $env:UV_INSTALL_DIR -Recurse -Filter 'uv.exe' | Select-Object -First 1
    }
    if (-not $uv) { throw 'uv was not installed' }
    $uv = $uv.FullName

    Step 'Installing Python 3.12'
    $venv = Join-Path $Dir '.venv'
    $py   = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $py)) {
        Run $uv @('venv', '--python', '3.12', $venv)
    }

    Step 'Installing Python packages (the largest download)'
    Run $uv @('pip', 'install', '--python', $py, '-r', (Join-Path $Dir 'requirements.txt'))
    # insightface would pull in the CPU-only onnxruntime and break GPU inference;
    # its real dependencies are already in requirements.txt.
    Run $uv @('pip', 'install', '--python', $py, '--no-deps', 'insightface>=2.1')

    Push-Location $Dir
    try {
        $env:PYTHONIOENCODING = 'utf-8'

        Step 'Downloading the models'
        Run $py @('download_models.py')

        Step 'Checking that the models run on the graphics card'
        Run $py @('tools\check_install.py')
    } finally {
        Pop-Location
    }

    Step 'Creating desktop shortcuts'
    $desktop = [Environment]::GetFolderPath('Desktop')
    $shell = New-Object -ComObject WScript.Shell
    foreach ($item in @(@('liveswap', 'run.bat'), @('liveswap test', 'test.bat'))) {
        $shortcut = $shell.CreateShortcut((Join-Path $desktop ($item[0] + '.lnk')))
        $shortcut.TargetPath = Join-Path $Dir $item[1]
        $shortcut.WorkingDirectory = $Dir
        $shortcut.Save()
    }

    Write-Host ''
    Write-Host 'Done.' -ForegroundColor Green
    Write-Host 'On the desktop:'
    Write-Host '  "liveswap"       starts the app'
    Write-Host '  "liveswap test"  runs a 30 second speed test and saves a log file to the desktop'
} catch {
    Write-Host ''
    Write-Host "INSTALL FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please send this file to the developer: $Log" -ForegroundColor Yellow
} finally {
    try { Stop-Transcript | Out-Null } catch { }
    Read-Host 'Press Enter to close'
}
