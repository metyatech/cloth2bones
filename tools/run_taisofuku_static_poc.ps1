[CmdletBinding()]
param(
    [string]$BlenderExe,
    [string]$SourceBlend,
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($BlenderExe)) {
    $downloadRoot = Join-Path (Join-Path 'D:\' 'Users') 'Origin\Downloads'
    $BlenderExe = Join-Path $downloadRoot 'cloth_poc_out\tools\blender-5.2.0-windows-x64\blender.exe'
}
if ([string]::IsNullOrWhiteSpace($SourceBlend)) {
    if ([string]::IsNullOrWhiteSpace($downloadRoot)) {
        $downloadRoot = Join-Path (Join-Path 'D:\' 'Users') 'Origin\Downloads'
    }
    $SourceBlend = Join-Path $downloadRoot 'RyuonTaisofuku_physbone_test.blend'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    if ([string]::IsNullOrWhiteSpace($downloadRoot)) {
        $downloadRoot = Join-Path (Join-Path 'D:\' 'Users') 'Origin\Downloads'
    }
    $OutputRoot = Join-Path $downloadRoot 'cloth_poc_out\taisofuku_static_review'
}

if (-not (Test-Path -LiteralPath $BlenderExe -PathType Leaf)) {
    throw "Blender executable does not exist: $BlenderExe"
}
if (-not (Test-Path -LiteralPath $SourceBlend -PathType Leaf)) {
    throw "Source Blend does not exist: $SourceBlend"
}

$builder = Join-Path $PSScriptRoot 'build_taisofuku_static_poc.py'
$verifier = Join-Path $PSScriptRoot 'verify_taisofuku_static_poc.py'
$report = Join-Path $OutputRoot 'static_equilibrium_report.json'
$reviewBlend = Join-Path $OutputRoot 'Taisofuku_StaticTeacher_Review.blend'

function Invoke-BlenderScript {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $stdout = Join-Path ([System.IO.Path]::GetTempPath()) ("cloth2bones_static_" + $Label + '_stdout.log')
    $stderr = Join-Path ([System.IO.Path]::GetTempPath()) ("cloth2bones_static_" + $Label + '_stderr.log')
    $process = Start-Process -FilePath $BlenderExe -ArgumentList (@('--background', '--factory-startup', '-noaudio', '--python', $Script, '--') + $Arguments) -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $errorLines = @(Get-Content -LiteralPath $stderr | Select-String -Pattern 'Traceback|Error:|Exception:')
    if ($process.ExitCode -ne 0 -or $errorLines.Count -gt 0) {
        Write-Output "[$Label] stdout:"
        Get-Content -LiteralPath $stdout
        Write-Output "[$Label] stderr:"
        Get-Content -LiteralPath $stderr
        throw "Blender $Label failed with exit code $($process.ExitCode)"
    }
    Get-Content -LiteralPath $stdout
}

Invoke-BlenderScript -Script $builder -Arguments @('--source', $SourceBlend, '--output-root', $OutputRoot) -Label 'build'
if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
    throw "Static equilibrium report was not created: $report"
}
if (-not (Test-Path -LiteralPath $reviewBlend -PathType Leaf)) {
    throw "Review Blend was not created: $reviewBlend"
}

Invoke-BlenderScript -Script $verifier -Arguments @('--source', $SourceBlend, '--blend', $reviewBlend, '--report', $report) -Label 'verify'
Write-Output 'PASS: Taisofuku static review Blend generated for user visual inspection.'
