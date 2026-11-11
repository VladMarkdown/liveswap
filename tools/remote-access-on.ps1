# Opens key-only SSH access to this PC for the liveswap developer, reachable
# only over Tailscale. Run in an ADMINISTRATOR PowerShell. Undo it with
# remote-access-off.ps1.
#
# What it changes on this PC:
#   - installs and starts the built-in Windows OpenSSH Server
#   - turns SSH password logins off (key only)
#   - adds ONE public key, marked "liveswap-test", for this Windows account
#   - limits the SSH firewall rule to Tailscale addresses (100.64.0.0/10)
#   - installs Tailscale if it is missing
# ASCII-only on purpose (Windows PowerShell 5.1).

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$PublicKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPKFm+CmvuPavWhoT7qN9JXLneyA2AdN8zXyuXM+VTWl liveswap-test'

function Step($text) { Write-Host ''; Write-Host "==> $text" -ForegroundColor Cyan }

Step 'Installing the Windows OpenSSH Server'
$capability = Get-WindowsCapability -Online -Name 'OpenSSH.Server*' | Select-Object -First 1
if ($capability.State -ne 'Installed') {
    Add-WindowsCapability -Online -Name $capability.Name | Out-Null
}
Start-Service sshd          # the first start creates sshd_config and the host keys
Set-Service -Name sshd -StartupType Automatic

Step 'Allowing key logins only'
$config = Join-Path $env:ProgramData 'ssh\sshd_config'
$lines = Get-Content $config | Where-Object { $_ -notmatch '^\s*#?\s*PasswordAuthentication\s' }
Set-Content -Path $config -Value (@('PasswordAuthentication no') + $lines) -Encoding ascii

Step 'Adding the developer key'
# Windows reads keys for administrator accounts from one shared file and for
# other accounts from the user's profile. Write both so either case works.
# Groups are named by SID: their display names are translated on non-English Windows.
$adminKeys = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
$userDir   = Join-Path $env:USERPROFILE '.ssh'
$userKeys  = Join-Path $userDir 'authorized_keys'
New-Item -ItemType Directory -Force -Path $userDir | Out-Null
foreach ($file in @($adminKeys, $userKeys)) {
    $existing = if (Test-Path $file) { @(Get-Content $file | Where-Object { $_ -notmatch 'liveswap-test' }) } else { @() }
    Set-Content -Path $file -Value ($existing + $PublicKey) -Encoding ascii
}
& icacls.exe $adminKeys /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null

Step 'Limiting SSH to Tailscale addresses'
$rule = Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue
if ($rule) {
    Set-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -Enabled True -RemoteAddress '100.64.0.0/10'
} else {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (Tailscale only)' `
        -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow -RemoteAddress '100.64.0.0/10' | Out-Null
}
Restart-Service sshd

Step 'Installing Tailscale'
$tailscale = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
if (-not (Test-Path $tailscale)) {
    $msi = Join-Path $env:TEMP 'tailscale-setup.msi'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://pkgs.tailscale.com/stable/tailscale-setup-latest-amd64.msi' -OutFile $msi
    $install = Start-Process msiexec.exe -ArgumentList @('/i', "`"$msi`"", '/qn', '/norestart') -Wait -PassThru
    if ($install.ExitCode -ne 0) { throw "Tailscale install failed with exit code $($install.ExitCode)" }
}

$ip = ''
try { $ip = (& $tailscale ip -4 2>$null | Select-Object -First 1) } catch { }

Write-Host ''
Write-Host 'SSH access is ready.' -ForegroundColor Green
if (-not $ip) {
    Write-Host 'Tailscale is installed but not signed in yet.' -ForegroundColor Yellow
    Write-Host 'Sign in with the key the developer sent you:'
    Write-Host "  & `"$tailscale`" up --unattended --authkey <KEY-FROM-DEVELOPER>"
    Write-Host 'Then run this to see the address:'
    Write-Host "  & `"$tailscale`" ip -4"
}
Write-Host ''
Write-Host 'Send these two values to the developer:'
Write-Host "  user: $env:USERNAME"
Write-Host "  ip:   $ip"
