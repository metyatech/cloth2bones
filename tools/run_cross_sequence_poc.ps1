[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ReferenceNpz,
    [Parameter(Mandatory = $true)][string]$CommonPoseNpz,
    [Parameter(Mandatory = $true)][string]$SyntheticRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [int]$Frames = 120
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$generator = Join-Path $repoRoot 'tools\generate_cross_sequence_synthetic.py'
$audit = Join-Path $repoRoot 'tools\audit_cross_sequence_dataset.py'
$analysis = Join-Path $repoRoot 'tools\cross_sequence_poc.py'
$plot = Join-Path $repoRoot 'tools\plot_cross_sequence_results.py'

foreach ($required in @($ReferenceNpz, $CommonPoseNpz, $generator, $audit, $analysis, $plot)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required input is missing: $required"
    }
}
New-Item -ItemType Directory -Force -Path $SyntheticRoot | Out-Null
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

Invoke-HeadlessTool -FilePath 'python' -Arguments @(
    $generator, '--reference-npz', $ReferenceNpz, '--poses', $CommonPoseNpz,
    '--out-root', $SyntheticRoot, '--frames', [string]$Frames
) -Name 'generate_cross_sequence_synthetic' -ExpectedOutputs @((Join-Path $SyntheticRoot 'manifest.json'))

Invoke-HeadlessTool -FilePath 'python' -Arguments @(
    $audit, '--root', $SyntheticRoot, '--out', (Join-Path $OutputRoot 'synthetic_inventory.json')
) -Name 'audit_synthetic_sequences' -ExpectedOutputs @((Join-Path $OutputRoot 'synthetic_inventory.json'))

Invoke-HeadlessTool -FilePath 'python' -Arguments @(
    $analysis, '--dataset-root', $SyntheticRoot, '--poses', $CommonPoseNpz,
    '--out-root', $OutputRoot, '--primary-test', 'seq_D_combined_holdout'
) -Name 'cross_sequence_analysis' -ExpectedOutputs @((Join-Path $OutputRoot 'cross_sequence_report.json'), (Join-Path $OutputRoot 'metrics.csv'))

Invoke-HeadlessTool -FilePath 'python' -Arguments @(
    $plot, '--report', (Join-Path $OutputRoot 'cross_sequence_report.json'),
    '--out', (Join-Path $OutputRoot 'plots')
) -Name 'cross_sequence_plots' -ExpectedOutputs @((Join-Path $OutputRoot 'plots\primary_per_frame_rms.svg'))

Write-Output "PASS: PoC 3.2 cross-sequence analysis completed. Report: $(Join-Path $OutputRoot 'cross_sequence_report.json')"
