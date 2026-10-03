# Niko Tracker - build the engine image: a slim, self-contained WSL distro as one .tar.xz.
# Author: Nihad Jiad Yousef.
#   powershell -ExecutionPolicy Bypass -File installer\engine\build_engine.ps1 [-Source Ubuntu-24.04] [-Out D:\NikoEngine]
# The working engine distro (-Source) is only read. Steps:
#   1. git archive HEAD            -> engine code of this version (committed files only)
#   2. stage.sh in -Source         -> slim root file system tar (no footage, solves, models, CUDA toolkit, secrets)
#   3. wsl --import NikoEngineBuild (a throw-away copy) and prepare.sh in it
#   4. wsl --export, xz (all cores) -> <Out>\niko-engine-<version>.tar.xz + .sha256; the copy is removed
# -ReuseStage: keep a complete build\stage.tar from an earlier run (step 2 is the slow one)
param([string]$Source = "Ubuntu-24.04", [string]$Out = "D:\NikoEngine", [switch]$ReuseStage)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$init = Get-Content (Join-Path $repo "addon\niko_tracker\__init__.py") -Raw
if ($init -notmatch '"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)') { throw "add-on version not found" }
$ver = "$($Matches[1]).$($Matches[2]).$($Matches[3])"
$build = Join-Path $Out "build"
$name = "NikoEngineBuild"

function To-Wsl([string]$p) { "/mnt/" + $p.Substring(0, 1).ToLower() + $p.Substring(2).Replace("\", "/") }
function Wsl([string[]]$a) {
    & wsl.exe @a
    if ($LASTEXITCODE -ne 0) { throw "wsl $($a -join ' ') failed ($LASTEXITCODE)" }
}
function Step([string]$m) { Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) }

New-Item -ItemType Directory -Force $build | Out-Null
$distros = (& wsl.exe --list --quiet) -replace "`0", "" | Where-Object { $_.Trim() }
if ($distros -contains $name) { Step "removing the previous build copy"; Wsl @("--unregister", $name) }

Step "engine code $ver (git archive HEAD)"
& git -C $repo archive --format=tar -o (Join-Path $build "engine-src.tar") HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }

$stage = Join-Path $build "stage.tar"
if ($ReuseStage -and (Test-Path $stage)) {
    Step "reusing $stage"
} else {
    Step "slim root file system from $Source"
    Wsl @("-d", $Source, "-u", "root", "--exec", "bash", (To-Wsl (Join-Path $PSScriptRoot "stage.sh")), (To-Wsl $stage))
}

Step "import as $name"
Wsl @("--import", $name, (Join-Path $build "distro"), $stage, "--version", "2")
Remove-Item $stage

Step "prepare the copy"
Wsl @("-d", $name, "-u", "root", "--exec", "bash", (To-Wsl (Join-Path $PSScriptRoot "prepare.sh")),
      (To-Wsl (Join-Path $build "engine-src.tar")), $ver)
Wsl @("--terminate", $name)

Step "export"
$tar = Join-Path $Out "niko-engine-$ver.tar"
Wsl @("--export", $name, $tar)
Wsl @("--unregister", $name)

Step "compress (xz, all cores)"
Wsl @("-d", $Source, "--exec", "xz", "-T0", "-6", "-f", (To-Wsl $tar))
$xz = "$tar.xz"
$hash = (Get-FileHash $xz -Algorithm SHA256).Hash.ToLower()
"$hash  niko-engine-$ver.tar.xz" | Set-Content -Encoding ascii "$xz.sha256"
Remove-Item (Join-Path $build "engine-src.tar")
Step ("done: {0} ({1:N2} GB), sha256 {2}" -f $xz, ((Get-Item $xz).Length / 1GB), $hash)

Step "split into release parts (GitHub takes at most 2 GB per file)"
$base = "niko-engine-$ver.tar.xz"
Wsl @("-d", $Source, "--exec", "bash", "-c",
      "cd '$(To-Wsl $Out)' && rm -f $base.part* && split -b 1900M -d -a 2 $base $base.part && sha256sum $base.part* > parts.sha256")
$parts = Get-Content (Join-Path $Out "parts.sha256") | ForEach-Object {
    $h, $n = $_ -split '\s+\*?', 2
    [ordered]@{ name = $n.Trim(); sha256 = $h; size = (Get-Item (Join-Path $Out $n.Trim())).Length }
}
Remove-Item (Join-Path $Out "parts.sha256")
[ordered]@{ version = $ver; tar = $base; sha256 = $hash; size = (Get-Item $xz).Length; parts = @($parts) } |
    ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii (Join-Path $Out "niko-engine-$ver.parts.json")
# installer\build.ps1 reads the parts list from the repo: commit it with the release
Copy-Item (Join-Path $Out "niko-engine-$ver.parts.json") (Join-Path $PSScriptRoot "releases") -Force
Step ("{0} parts: {1}" -f @($parts).Count, (($parts | ForEach-Object { $_.name }) -join ", "))
