[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BlenderExe,
    [Parameter(Mandatory = $true)][string]$CommonPoseNpz,
    [Parameter(Mandatory = $true)][string]$TrianglesNpz,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [int]$Frames = 120,
    [int]$SettleFrames = 20
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$generator = Join-Path $repoRoot 'tools\generate_blender_physics_poc.py'
$analysis = Join-Path $repoRoot 'tools\physics_cross_sequence_poc.py'
$motions = @(
    'A_left_down', 'B_right_down', 'C_both_down', 'D_left_down_up',
    'E_right_down_up', 'F_alternating', 'G_diagonal', 'H_unseen_combined',
    'I_speed_variant', 'J_reverse_history'
)

foreach ($required in @($BlenderExe, $CommonPoseNpz, $TrianglesNpz, $generator, $analysis)) {
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
            throw "Expected output is missing after $Name : $expected"
        }
    }
}

foreach ($motion in $motions) {
    $teacher = Join-Path $OutputRoot "physics_sequences\$motion\teacher.npz"
    Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
        '--background', '--factory-startup', '-noaudio', '--python', $generator, '--',
        '--reference-poses', $CommonPoseNpz, '--triangles', $TrianglesNpz,
        '--out-root', $OutputRoot, '--motion', $motion,
        '--frames', [string]$Frames, '--settle', [string]$SettleFrames
    ) -Name ('generate_' + $motion) -ExpectedOutputs @($teacher)
}

Invoke-HeadlessTool -FilePath 'python' -Arguments @(
    $analysis, '--dataset-root', $OutputRoot, '--common-poses', $CommonPoseNpz,
    '--triangles', $TrianglesNpz, '--out-root', $OutputRoot,
    '--primary-test', 'H_unseen_combined'
) -Name 'physics_cross_sequence_analysis' -ExpectedOutputs @(
    (Join-Path $OutputRoot 'dataset_manifest.json'),
    (Join-Path $OutputRoot 'physics_cross_sequence_report.json'),
    (Join-Path $OutputRoot 'model_metrics.csv')
)

Write-Output "PASS: Blender physics cross-sequence PoC completed. Report: $(Join-Path $OutputRoot 'physics_cross_sequence_report.json')"
