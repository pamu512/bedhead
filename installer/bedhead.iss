; Inno Setup script for the bedhead Windows installer.
; Build: ISCC.exe installer\bedhead.iss  (after pyinstaller bedhead.spec)
; Output: dist\bedhead-<version>-windows-x64.exe

#define MyAppName "Bedhead"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Anoop Pamu"
#define MyAppExeName "bedhead.exe"
#define MyAppURL "https://github.com/pamu512/bedhead"

[Setup]
AppId={{8E4C2A7B-9F3D-4E56-B1A8-2C5D7E9F0A11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=dist
OutputBaseFilename=bedhead-{#MyAppVersion}-windows-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; video-call tool: highest available privileges, no UAC dance for per-user
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
; the one-file PyInstaller build
Source: "dist\bedhead.exe"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; add {app} to the user PATH so `bedhead` works from any terminal
Name: "{app}"; Flags: uninsneveruninstall

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}"; Check: NeedsAddPath('{app}'); Flags: preservestringtype

[Icons]
Name: "{group}\Bedhead CLI"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall Bedhead"; Filename: "{uninstallexe}"

[Messages]
WelcomeLabel2=This will install Bedhead, the good-day camera for video calls.%n%nAll processing happens on your machine. Nothing is uploaded, ever.%n%nThe installer adds `bedhead` to your PATH (new terminals only).%n%nModels (~20 MiB) download once into your local cache on first run.

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
  AppDir: string;
begin
  AppDir := ExpandConstant(Param);
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(AppDir) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    Log('Installed bedhead to ' + ExpandConstant('{app}'));
end;
