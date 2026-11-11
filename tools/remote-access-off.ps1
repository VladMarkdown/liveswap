# Closes the SSH access opened by remote-access-on.ps1.
# Run in an ADMINISTRATOR PowerShell. ASCII-only on purpose.
#
# Removes the "liveswap-test" key, then stops and disables the OpenSSH Server.
# Tailscale is left installed but signed out; uninstall it from
# "Apps" in Windows Settings if it is no longer wanted.

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

foreach ($file in @((Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'),
                    (Join-Path $env:USERPROFILE '.ssh\authorized_keys'))) {
    if (Test-Path $file) {
        $kept = @(Get-Content $file | Where-Object { $_ -notmatch 'liveswap-test' })
        Set-Content -Path $file -Value $kept -Encoding ascii
    }
}

Stop-Service sshd -ErrorAction SilentlyContinue
Set-Service -Name sshd -StartupType Disabled
Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue |
    Set-NetFirewallRule -Enabled False

$tailscale = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
if (Test-Path $tailscale) {
    try { & $tailscale logout | Out-Null } catch { }
}

Write-Host 'Remote access is closed: key removed, SSH server stopped and disabled.' -ForegroundColor Green
