[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Npz,
    [Parameter(Mandatory = $true)][string]$CleanFbx,
    [Parameter(Mandatory = $true)][string]$BlenderExe,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [ValidateSet('linear', 'ridge', 'polynomial', 'nearest', 'rbf', 'local-ridge')][string]$SelectedModel = 'nearest',
    [string]$BoneCounts = '50,32,20,16,8'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$analysis = Join-Path $repoRoot 'tools\body_local_poc.py'
$dump = Join-Path $repoRoot 'tools\dump_blender_rig_poses.py'
$apply = Join-Path $repoRoot 'tools\apply_body_driven_poses.py'
$verify = Join-Path $repoRoot 'tools\verify_body_local_fbx.py'
$render = Join-Path $repoRoot 'tools\render_body_local_comparison.py'
$poses = Join-Path $OutputRoot 'clean_rig_poses.npz'
$report = Join-Path $OutputRoot 'body_local_report.json'
$teacherFbx = Join-Path $OutputRoot 'body_local_teacher_50.fbx'
$predictedFbx = Join-Path $OutputRoot 'body_local_predicted_50.fbx'

foreach ($required in @($Npz, $CleanFbx, $BlenderExe, $analysis, $dump, $apply, $verify, $render)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required input is missing: $required"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Invoke-HeadlessTool {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$ExpectedOutputs = @()
    )
    $stdout = Join-Path $OutputRoot ($Name + '.stdout.log')
    $stderr = Join-Path $OutputRoot ($Name + '.stderr.log')
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $errorText = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
    if ($process.ExitCode -ne 0 -or $errorText -match 'Traceback \(most recent call last\)|RuntimeError:|ValueError:') {
        throw "Tool failed: $Name. See $stdout and $stderr"
    }
    foreach ($expected in $ExpectedOutputs) {
        if (-not (Test-Path -LiteralPath $expected -PathType Leaf)) {
            throw "Expected output is missing after $Name`: $expected"
        }
    }
}

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', $dump, '--',
    '--fbx', $CleanFbx, '--out', $poses, '--start', '1', '--end', '240'
) -Name 'dump_clean_rig_poses' -ExpectedOutputs @($poses)

Invoke-HeadlessTool -FilePath 'python' -Arguments @(
    $analysis, '--npz', $Npz, '--poses', $poses, '--out-root', $OutputRoot,
    '--selected-model', $SelectedModel, '--bone-counts', $BoneCounts
) -Name 'body_local_analysis' -ExpectedOutputs @($report, (Join-Path $OutputRoot 'predicted_local_poses_50.npz'))

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', $apply, '--',
    '--fbx', $CleanFbx, '--poses', (Join-Path $OutputRoot 'teacher_local_poses_50.npz'), '--out', $teacherFbx,
    '--expected', (Join-Path $OutputRoot 'body_local_features.npz'), '--report', (Join-Path $OutputRoot 'teacher_preexport.json')
) -Name 'apply_teacher' -ExpectedOutputs @($teacherFbx)

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', $apply, '--',
    '--fbx', $CleanFbx, '--poses', (Join-Path $OutputRoot 'predicted_local_poses_50.npz'), '--out', $predictedFbx,
    '--expected', (Join-Path $OutputRoot 'body_local_features.npz'), '--report', (Join-Path $OutputRoot 'predicted_preexport.json')
) -Name 'apply_prediction' -ExpectedOutputs @($predictedFbx)

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', $verify, '--',
    '--fbx', $predictedFbx, '--teacher', (Join-Path $OutputRoot 'body_local_features.npz'),
    '--json', (Join-Path $OutputRoot 'predicted_verify.json')
) -Name 'verify_prediction' -ExpectedOutputs @((Join-Path $OutputRoot 'predicted_verify.json'))

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', $render, '--',
    '--teacher-fbx', $teacherFbx, '--predicted-fbx', $predictedFbx,
    '--out', (Join-Path $OutputRoot 'previews'), '--blend', (Join-Path $OutputRoot 'body_local_comparison.blend'),
    '--frames', '1,61,121,181,240'
) -Name 'render_comparison' -ExpectedOutputs @((Join-Path $OutputRoot 'previews\body_local_comparison_181.png'))

Write-Output "PASS: body-local PoC 3.1 completed. Report: $report"
Write-Output "Predicted FBX: $predictedFbx"
