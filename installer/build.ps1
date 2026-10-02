# Build the Windows installer: installer\Output\NikoTracker-Setup-<version>.exe
# Niko Tracker - author: Nihad Jihad ("Niko").
# Needs Inno Setup 6 (ISCC.exe; per-user install in %LOCALAPPDATA%\Programs\Inno Setup 6).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found (ISCC.exe)" }
# the version is the add-on's (bl_info "version": (0, 3, 5))
$init = Get-Content (Join-Path $here "..\addon\niko_tracker\__init__.py") -Raw
if ($init -notmatch '"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)') { throw "add-on version not found" }
$version = "$($Matches[1]).$($Matches[2]).$($Matches[3])"
& $iscc "/DAppVersion=$version" (Join-Path $here "NikoTracker.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
Get-Item (Join-Path $here "Output\NikoTracker-Setup-$version.exe") | Select-Object Name, Length
