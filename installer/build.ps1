# Build the Windows installer: installer\Output\NikoTracker-Setup-<version>.exe
# Niko Tracker - author: Nihad Jiad Yousef.
# Needs Inno Setup 6 (ISCC.exe; per-user install in %LOCALAPPDATA%\Programs\Inno Setup 6).
# The engine it downloads: the newest installer\engine\releases\niko-engine-<v>.parts.json with
# v <= this version (build_engine.ps1 writes them; a code-only release keeps the last image).
#   -EngineBase  where the installer downloads the parts from (default: that image's GitHub release)
#   -NoEngine    an installer without the engine steps (add-on and hardware check only)
param([string]$EngineBase = "", [switch]$NoEngine)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found (ISCC.exe)" }
# the version is the add-on's (bl_info "version")
$init = Get-Content (Join-Path $here "..\addon\niko_tracker\__init__.py") -Raw
if ($init -notmatch '"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)') { throw "add-on version not found" }
$version = "$($Matches[1]).$($Matches[2]).$($Matches[3])"

# engine_parts.iss: the engine image this installer downloads (generated, not in git)
function VersionKey([string]$v) { [version]$v }
$parts = Get-ChildItem (Join-Path $here "engine\releases") -Filter "niko-engine-*.parts.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^niko-engine-(\d+\.\d+\.\d+)\.parts\.json$' } |
    Where-Object { (VersionKey ($_.Name -replace '^niko-engine-|\.parts\.json$', '')) -le (VersionKey $version) } |
    Sort-Object { VersionKey ($_.Name -replace '^niko-engine-|\.parts\.json$', '') } |
    Select-Object -Last 1 -ExpandProperty FullName
$inc = Join-Path $here "engine_parts.iss"
if ($NoEngine -or -not $parts) {
    if (-not $NoEngine) { Write-Warning "no engine image <= $version in installer\engine\releases: building without the engine steps" }
    @"
const
  EngineVersion = '';
  EngineBaseUrl = '';
  EngineSizeText = '0 GB';
  EngineTarName = '';

procedure AddEngineParts(Page: TDownloadWizardPage; const Base: String);
begin
end;

function JoinEngineParts(const Tar: String): Boolean;
begin
  Result := False;
end;
"@ | Set-Content -Encoding ascii $inc
} else {
    $p = Get-Content $parts -Raw | ConvertFrom-Json
    if (-not $EngineBase) { $EngineBase = "https://github.com/nihadniko9-dev/niko-tracker/releases/download/v$($p.version)/" }
    $adds = ($p.parts | ForEach-Object { "  Page.Add(Base + '$($_.name)', '$($_.name)', '$($_.sha256)');" }) -join "`r`n"
    $join = ($p.parts | ForEach-Object { "'`"' + T + '$($_.name)`"'" }) -join " + '+' + "
    $dels = ($p.parts | ForEach-Object { "  DeleteFile(T + '$($_.name)');" }) -join "`r`n"
    @"
const
  EngineVersion = '$($p.version)';
  EngineBaseUrl = '$EngineBase';
  EngineSizeText = '$([math]::Round($p.size / 1GB, 1)) GB';
  EngineTarName = '$($p.tar)';

procedure AddEngineParts(Page: TDownloadWizardPage; const Base: String);
begin
$adds
end;

// the parts back into one file (cmd copy /b), then the parts are deleted
function JoinEngineParts(const Tar: String): Boolean;
var
  T: String;
  Code: Integer;
begin
  T := ExpandConstant('{tmp}\');
  Result := Exec(ExpandConstant('{cmd}'), '/C copy /b ' + $join + ' "' + Tar + '"', T, SW_HIDE,
                 ewWaitUntilTerminated, Code) and (Code = 0) and FileExists(Tar);
$dels
end;
"@ | Set-Content -Encoding ascii $inc
    Write-Host "engine $($p.version): $(@($p.parts).Count) parts from $EngineBase"
}

& $iscc "/DAppVersion=$version" (Join-Path $here "NikoTracker.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
Get-Item (Join-Path $here "Output\NikoTracker-Setup-$version.exe") | Select-Object Name, Length
