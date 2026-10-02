; Niko Tracker - Windows installer (Inno Setup 6).
; Niko Tracker Engine - author: Nihad Jihad ("Niko"). Personal, non-commercial.
;
; Build:  ISCC.exe installer\NikoTracker.iss   ->  installer\Output\NikoTracker-Setup.exe
; The EXE is small: the engine (a prebuilt WSL2 Ubuntu image with the Python environments, ~6 GB
; compressed) and the models (~13 GB, from their official sources; SAM 3 with the user's own
; Hugging Face login) are downloaded during setup, resumable, with checksums.
;
; Steps the wizard runs:
;   1. hardware_check.ps1  - Windows, NVIDIA GPU / driver / VRAM, RAM, disk, WSL -> profile or stop
;   2. WSL2 on             - `wsl --install --no-distribution` when missing (restart, setup resumes)
;   3. engine              - download niko-engine.tar.zst, `wsl --import NikoEngine <dir> <tar>`
;   4. models              - fetch_checkpoints.py inside the engine (sizes shown, resumable)
;   5. Blender add-on      - install_addon.ps1 (every Blender found, enabled)
;   6. check               - `niko doctor` + a 30-frame smoke solve
;
;
; BUILT 2026-10-02 with Inno Setup 6.7.3 (installer\build.ps1). What this EXE does today: the
; hardware check page (stops on blockers), the Blender add-on into every Blender found, the docs,
; a Start menu entry and an uninstaller. Steps 2-4 and 6 (WSL, engine image, models, smoke solve)
; still need the engine image (a slim distro without the benchmark data) and a place to host it
; (Nihad's choice); until then the engine is set up by hand (README).

#define AppName "Niko Tracker"
#ifndef AppVersion
  #define AppVersion "0.3.7"
#endif
#define AppPublisher "Nihad Jihad (Niko)"
#define EngineURL "https://REPLACE-WITH-HOSTING/niko-engine-0.2.0.tar.zst"
#define EngineSHA256 "REPLACE-WITH-CHECKSUM"

[Setup]
AppId={{6A33BBFA-634D-4DE8-9A4E-C853FD59BE42}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Niko Tracker
; per user (no administrator prompt): the add-on, WSL distros and the engine all live per user
PrivilegesRequired=lowest
DefaultGroupName=Niko Tracker
OutputBaseFilename=NikoTracker-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; artwork: installer\art\make_art.py (100 % and 200 % sizes for high-DPI screens)
WizardImageFile=art\wizard_side_1x.bmp,art\wizard_side_2x.bmp
WizardSmallImageFile=art\wizard_small_1x.bmp,art\wizard_small_2x.bmp
SetupIconFile=art\niko.ico
UninstallDisplayIcon={app}\niko.ico
UninstallDisplayName={#AppName} {#AppVersion}
LicenseFile=..\LICENSE
AppCopyright=Copyright (c) 2026 Nihad Jihad (Niko). All rights reserved.
InfoAfterFile=after_install.txt
OutputDir=Output

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Components]
Name: "blender"; Description: "Blender add-on (the Niko track tab)"; Types: full custom; Flags: fixed

[Files]
Source: "..\addon\niko_tracker\*.py"; DestDir: "{app}\addon\niko_tracker"; Components: blender
Source: "hardware_check.ps1"; DestDir: "{app}\installer"
Source: "install_addon.ps1"; DestDir: "{app}\installer"
Source: "..\README.md"; DestDir: "{app}"
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"
Source: "..\docs\LICENSES.md"; DestDir: "{app}\docs"
Source: "art\niko.ico"; DestDir: "{app}"

[Icons]
Name: "{group}\Niko Tracker - read me"; Filename: "{app}\README.md"; IconFilename: "{app}\niko.ico"
Name: "{group}\Uninstall Niko Tracker"; Filename: "{uninstallexe}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\install_addon.ps1"" -Source ""{app}\addon\niko_tracker"""; StatusMsg: "Installing the Blender add-on..."; Components: blender; Flags: runhidden waituntilterminated

[Code]
var
  HwPage: TOutputMsgMemoWizardPage;
  HwOk: Boolean;

function RunCapture(const Exe, Params: String; var Output: AnsiString): Integer;
var
  Tmp: String;
  Code: Integer;
begin
  Tmp := ExpandConstant('{tmp}\niko_out.txt');
  Exec(ExpandConstant('{cmd}'), '/C ""' + Exe + '" ' + Params + ' > "' + Tmp + '" 2>&1"', '', SW_HIDE,
       ewWaitUntilTerminated, Code);
  LoadStringFromFile(Tmp, Output);
  Result := Code;
end;

procedure InitializeWizard;
begin
  HwPage := CreateOutputMsgMemoPage(wpWelcome, 'Checking this computer',
    'GPU, driver, memory, disk and WSL2', 'Niko Tracker needs an NVIDIA GPU with 8 GB or more.', '');
end;

procedure CurPageChanged(CurPageID: Integer);
var
  Json: AnsiString;
begin
  if CurPageID = HwPage.ID then
  begin
    ExtractTemporaryFile('hardware_check.ps1');
    RunCapture('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' +
               ExpandConstant('{tmp}\hardware_check.ps1') + '"', Json);
    HwPage.RichEditViewer.Lines.Text := String(Json);
    // Windows PowerShell writes "ok":  true (two spaces), PowerShell 7 one
    HwOk := (Pos('"ok":  true', String(Json)) > 0) or (Pos('"ok": true', String(Json)) > 0);
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = HwPage.ID) and (not HwOk) then
  begin
    MsgBox('This computer cannot run the tracking engine yet. The "blockers" list above says why.',
           mbError, MB_OK);
    Result := False;
  end;
end;
