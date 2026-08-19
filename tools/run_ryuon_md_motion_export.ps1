[CmdletBinding()]
param(
    [string]$BlenderExe,
    [string]$SourceBlend,
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($SourceBlend)) {
    $SourceBlend = Join-Path (Join-Path 'D:\' 'Users') 'Origin\Downloads\RyuonTaisofuku_physbone_test.blend'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (Join-Path 'D:\' 'Users') 'Origin\Downloads\cloth_poc_out\taisofuku_md_teacher'
}

if ([string]::IsNullOrWhiteSpace($BlenderExe)) {
    $candidates = @(Get-ChildItem -LiteralPath 'C:\Program Files\Blender Foundation' -Filter 'blender.exe' -File -Recurse -ErrorAction SilentlyContinue)
    if ($candidates.Count -ne 1) {
        $paths = $candidates | ForEach-Object { $_.FullName }
        throw "Blender executable was not specified and search found $($candidates.Count) candidates: $($paths -join ', ')"
    }
    $BlenderExe = $candidates[0].FullName
}

if (-not (Test-Path -LiteralPath $BlenderExe -PathType Leaf)) {
    throw "Blender executable does not exist: $BlenderExe"
}
if (-not (Test-Path -LiteralPath $SourceBlend -PathType Leaf)) {
    throw "Source Blend does not exist: $SourceBlend"
}

$generator = Join-Path $PSScriptRoot 'export_ryuon_md_motion.py'
$verifier = Join-Path $PSScriptRoot 'verify_ryuon_md_motion.py'
$motionRoot = Join-Path $OutputRoot 'motion'
$motionFiles = @(
    (Join-Path $motionRoot 'Ryuon_MD_LeftDown.fbx'),
    (Join-Path $motionRoot 'Ryuon_MD_RightDown.fbx'),
    (Join-Path $motionRoot 'Ryuon_MD_BothDown.fbx')
)
$verifyReport = Join-Path $OutputRoot 'ryuon_md_motion_verify.json'

function Invoke-BlenderScript {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $stdout = Join-Path ([System.IO.Path]::GetTempPath()) ("cloth2bones_" + $Label + '_stdout.log')
    $stderr = Join-Path ([System.IO.Path]::GetTempPath()) ("cloth2bones_" + $Label + '_stderr.log')
    $process = Start-Process -FilePath $BlenderExe -ArgumentList (@('--background', '--factory-startup', '-noaudio', '--python', $Script, '--') + $Arguments) -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $errorLines = @(Get-Content -LiteralPath $stderr | Select-String -Pattern 'Traceback|Error:|Exception:')
    if ($process.ExitCode -ne 0 -or $errorLines.Count -gt 0) {
        Write-Output "[$Label] stdout:"
        Get-Content -LiteralPath $stdout
        Write-Output "[$Label] stderr:"
        Get-Content -LiteralPath $stderr
        throw "Blender $Label failed with exit code $($process.ExitCode)"
    }
}

Invoke-BlenderScript -Script $generator -Arguments @('--source', $SourceBlend, '--output-root', $OutputRoot) -Label 'generate'
foreach ($motionFile in $motionFiles) {
    if (-not (Test-Path -LiteralPath $motionFile -PathType Leaf)) {
        throw "Expected motion FBX was not created: $motionFile"
    }
}
Invoke-BlenderScript -Script $verifier -Arguments @('--source', $SourceBlend, '--fbx', $motionFiles[0], '--fbx', $motionFiles[1], '--fbx', $motionFiles[2], '--output', $verifyReport) -Label 'verify'

$verify = Get-Content -LiteralPath $verifyReport -Raw | ConvertFrom-Json
if (-not $verify.pass) {
    throw "FBX verification report did not pass: $verifyReport"
}
Write-Output 'PASS: Ryuon Marvelous Designer motion FBXs verified.'
