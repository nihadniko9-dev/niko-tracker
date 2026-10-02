# Niko Tracker - install (or update) the Blender add-on into every installed Blender and enable it.
# Niko Tracker Engine - author: Nihad Jihad ("Niko").
#   powershell -ExecutionPolicy Bypass -File installer\install_addon.ps1 [-Source <addon\niko_tracker>] [-NoEnable]
# For each "Blender x.y" in Program Files: copies the add-on to
# %APPDATA%\Blender Foundation\Blender\x.y\scripts\addons\niko_tracker (older copy replaced), then
# runs that Blender once in the background to enable it and save the preferences (so the
# "Niko track" tab is there on the next start).
param([string]$Source = "", [switch]$NoEnable)

$ErrorActionPreference = "Stop"
if (-not $Source) { $Source = Join-Path (Split-Path -Parent $PSScriptRoot) "addon\niko_tracker" }
if (-not (Test-Path (Join-Path $Source "__init__.py"))) { throw "No add-on at $Source" }
$version = (Select-String -Path (Join-Path $Source "__init__.py") -Pattern '"version":\s*\(([^)]*)\)').Matches[0].Groups[1].Value -replace "\s", "" -replace ",", "."

$root = Join-Path $env:ProgramFiles "Blender Foundation"
$done = @()
if (Test-Path $root) {
    foreach ($dir in Get-ChildItem $root -Directory) {
        $exe = Join-Path $dir.FullName "blender.exe"
        if (-not (Test-Path $exe)) { continue }
        if ($dir.Name -notmatch "(\d+\.\d+)") { continue }
        $ver = $Matches[1]
        $dst = Join-Path $env:APPDATA "Blender Foundation\Blender\$ver\scripts\addons\niko_tracker"
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        New-Item -ItemType Directory -Force $dst | Out-Null
        Copy-Item (Join-Path $Source "*.py") $dst
        $enabled = $false
        if (-not $NoEnable) {
            $py = "import addon_utils, bpy; addon_utils.enable('niko_tracker', default_set=True, persistent=True); bpy.ops.wm.save_userpref(); print('NIKO_ENABLED')"
            $outp = & $exe -b --python-expr $py 2>&1 | Out-String
            $enabled = $outp -match "NIKO_ENABLED"
        }
        $done += [ordered]@{ blender = $dir.Name; addon_dir = $dst; enabled = $enabled }
    }
}
[ordered]@{ addon_version = $version; installed = $done } | ConvertTo-Json -Depth 4
if ($done.Count -eq 0) { Write-Warning "No Blender found in $root" ; exit 1 }
