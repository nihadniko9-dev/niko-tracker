# Niko Tracker - hardware and system check for the installer (and for `niko doctor` on Windows).
# Niko Tracker Engine - author: Nihad Jihad.
#
# Prints one JSON object: what was found, a settings profile for this machine and the blockers.
#   powershell -ExecutionPolicy Bypass -File installer\hardware_check.ps1 [-Json path]
# Profile (from GPU memory; the engine reads it to size its work):
#   full   >= 24 GB  SAM 3 chunk 30 frames, CoTracker3 at 960x540, DA3 giant
#   medium 12-24 GB  SAM 3 chunk 15, CoTracker3 at 768x432
#   light   8-12 GB  SAM 3 chunk 8,  CoTracker3 at 640x360 (slower, same method)
#   none   no NVIDIA GPU with >= 8 GB: not supported (the models need CUDA)
param([string]$Json = "")

$ErrorActionPreference = "Continue"
$r = [ordered]@{ schema = "niko.hardware/1"; checked = (Get-Date).ToString("s") }
$blockers = New-Object System.Collections.ArrayList
$warnings = New-Object System.Collections.ArrayList

# Windows
$os = Get-CimInstance Win32_OperatingSystem
$r.windows = [ordered]@{ caption = $os.Caption; build = [int]$os.BuildNumber; version = $os.Version }
if ([int]$os.BuildNumber -lt 19045) { [void]$blockers.Add("Windows 10 22H2 (build 19045) or Windows 11 is needed for WSL2 with GPU") }

# memory and disk
$ramGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$r.ram_gb = $ramGB
if ($ramGB -lt 32) { [void]$warnings.Add("RAM $ramGB GB: 32 GB or more recommended (64 GB for 4K clips)") }
$r.disks = @(Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ne $null -and $_.Used -ne $null } |
    ForEach-Object { [ordered]@{ drive = $_.Name; free_gb = [math]::Round($_.Free / 1GB, 1) } })
$best = ($r.disks | Sort-Object { $_.free_gb } -Descending | Select-Object -First 1)
if ($best.free_gb -lt 60) { [void]$blockers.Add("No drive with 60 GB free (engine ~25 GB + working space)") }

# virtualisation / WSL
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$r.cpu = [ordered]@{ name = $cpu.Name.Trim(); cores = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors
                     virtualization_firmware = [bool]$cpu.VirtualizationFirmwareEnabled }
$wslExe = Get-Command wsl.exe -ErrorAction SilentlyContinue
$wsl = [ordered]@{ installed = [bool]$wslExe; version = ""; distros = @() }
if ($wslExe) {
    $v = (& wsl.exe --version 2>$null) -join "`n"
    $v = $v -replace "`0", ""
    if ($v -match "WSL[^:]*:\s*([0-9.]+)") { $wsl.version = $Matches[1] }
    $list = (& wsl.exe --list --quiet 2>$null) -join "`n"
    $wsl.distros = @(($list -replace "`0", "") -split "`r?`n" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })
}
$r.wsl = $wsl
if (-not $wsl.installed -or -not $wsl.version) { [void]$warnings.Add("WSL2 is not installed yet: the installer turns it on (needs a restart)") }

# NVIDIA GPU
$gpus = @()
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) {
    $rows = & nvidia-smi.exe --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits 2>$null
    foreach ($row in $rows) {
        $p = $row -split ",\s*"
        if ($p.Count -ge 4) {
            $gpus += [ordered]@{ name = $p[0]; vram_gb = [math]::Round([double]$p[1] / 1024, 1); driver = $p[2]; compute_capability = $p[3] }
        }
    }
}
$r.gpus = $gpus
$profile = "none"
if ($gpus.Count -eq 0) {
    [void]$blockers.Add("No NVIDIA GPU found (nvidia-smi): the tracking models need CUDA")
} else {
    $g = $gpus | Sort-Object { $_.vram_gb } -Descending | Select-Object -First 1
    $drv = [double](($g.driver -split "\.")[0])
    if ($drv -lt 570) { [void]$blockers.Add("NVIDIA driver $($g.driver): 570 or newer is needed (CUDA 12.8)") }
    $cc = [double]$g.compute_capability
    if ($cc -lt 7.5) { [void]$blockers.Add("GPU compute capability ${cc}: RTX 20 series (7.5) or newer is needed") }
    if ($g.vram_gb -ge 23) { $profile = "full" }
    elseif ($g.vram_gb -ge 11.5) { $profile = "medium" }
    elseif ($g.vram_gb -ge 7.5) { $profile = "light" }
    else { [void]$blockers.Add("GPU memory $($g.vram_gb) GB: at least 8 GB is needed") }
}
$settings = @{
    full   = [ordered]@{ sam3_chunk_frames = 30; cotracker_max_area = 518400; da3_model = "giant" }
    medium = [ordered]@{ sam3_chunk_frames = 15; cotracker_max_area = 331776; da3_model = "giant" }
    light  = [ordered]@{ sam3_chunk_frames = 8;  cotracker_max_area = 230400; da3_model = "giant" }
    none   = [ordered]@{}
}
$r.profile = $profile
$r.settings = $settings[$profile]

# Blender and After Effects
$bl = @()
foreach ($root in @("$env:ProgramFiles\Blender Foundation")) {
    if (Test-Path $root) { Get-ChildItem $root -Directory | ForEach-Object { if (Test-Path (Join-Path $_.FullName "blender.exe")) { $bl += $_.Name } } }
}
$r.blender = $bl
$ae = @()
if (Test-Path "$env:ProgramFiles\Adobe") {
    Get-ChildItem "$env:ProgramFiles\Adobe" -Directory -Filter "Adobe After Effects*" | ForEach-Object { $ae += $_.Name }
}
$r.after_effects = $ae
if ($bl.Count -eq 0) { [void]$warnings.Add("Blender not found in Program Files: install Blender 5.2, the add-on goes in afterwards") }

$r.blockers = @($blockers)
$r.warnings = @($warnings)
$r.ok = ($blockers.Count -eq 0)
$out = $r | ConvertTo-Json -Depth 6
if ($Json) { $out | Set-Content -Encoding UTF8 -Path $Json }
$out
