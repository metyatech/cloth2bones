[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DemBonesExe,
    [Parameter(Mandatory = $true)][string]$BlenderExe,
    [Parameter(Mandatory = $true)][string]$InputFbx,
    [Parameter(Mandatory = $true)][string]$InputAbc,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [ValidateRange(1, 256)][int]$BoneCount = 50,
    [int]$StartFrame = 1,
    [int]$EndFrame = 240,
    [double]$ExportGlobalScale = 0.01,
    [double]$ExportContentScale = 100.0,
    [double]$MaxRms = 0.05
)

$ErrorActionPreference = 'Stop'
$toolRoot = Join-Path $PSScriptRoot ''
$rawOutput = Join-Path $OutputRoot ("dem_raw_{0}.fbx" -f $BoneCount)
$demLog = Join-Path $OutputRoot ("dem_raw_{0}.log" -f $BoneCount)
$finalOutput = Join-Path $OutputRoot ("cloth_clean_rigid_{0}.fbx" -f $BoneCount)
$fitReport = Join-Path $OutputRoot ("cloth_clean_rigid_{0}_report.json" -f $BoneCount)
$blenderReport = Join-Path $OutputRoot ("cloth_clean_rigid_{0}_blender.json" -f $BoneCount)
$compareReport = Join-Path $OutputRoot ("cloth_clean_rigid_{0}_vs_abc.json" -f $BoneCount)

function Invoke-HeadlessTool {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $stderr = $LogPath + '.err'
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $LogPath -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        throw "Tool failed with exit code $($process.ExitCode): $FilePath. See $LogPath and $stderr"
    }
}

foreach ($required in @($DemBonesExe, $BlenderExe, $InputFbx, $InputAbc)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required input is missing: $required"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$comparisonScale = ($ExportGlobalScale * $ExportContentScale).ToString([System.Globalization.CultureInfo]::InvariantCulture)

Invoke-HeadlessTool -FilePath $DemBonesExe -Arguments @(
    ('-i=' + $InputFbx), ('-a=' + $InputAbc), ('-b=' + $BoneCount), ('-o=' + $rawOutput), ('--log=' + $demLog)
) -LogPath (Join-Path $OutputRoot 'dem_stdout.log')

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', (Join-Path $toolRoot 'build_clean_blender_rig.py'), '--',
    '--dem-fbx', $rawOutput, '--abc', $InputAbc, '--out', $finalOutput, '--report', $fitReport,
    '--start', $StartFrame, '--end', $EndFrame, '--passes', '8', '--export-global-scale', $ExportGlobalScale,
    '--export-content-scale', $ExportContentScale
) -LogPath (Join-Path $OutputRoot 'blender_fit.log')

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', (Join-Path $toolRoot 'blender_clean_output_verify.py'), '--',
    '--fbx', $finalOutput, '--json', $blenderReport, '--frames', "$StartFrame,$([int](($StartFrame + $EndFrame) / 2)),$EndFrame", '--expected-bones', $BoneCount
) -LogPath (Join-Path $OutputRoot 'blender_verify.log')

Invoke-HeadlessTool -FilePath $BlenderExe -Arguments @(
    '--background', '--factory-startup', '-noaudio', '--python', (Join-Path $toolRoot 'compare_fbx_to_abc.py'), '--',
    '--fbx', $finalOutput, '--abc', $InputAbc, '--json', $compareReport,
    '--frames', "$StartFrame,$([int](($StartFrame + $EndFrame) / 2)),$EndFrame", '--abc-scale', $comparisonScale
) -LogPath (Join-Path $OutputRoot 'blender_compare.log')

foreach ($requiredOutput in @($rawOutput, $finalOutput, $fitReport, $blenderReport, $compareReport)) {
    if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf)) {
        throw "Expected output is missing: $requiredOutput"
    }
}
$fit = Get-Content -LiteralPath $fitReport -Raw | ConvertFrom-Json
if ([int]$fit.bones -ne $BoneCount -or [int]$fit.zero_weight_vertices -ne 0) {
    throw "Clean rig acceptance failed: expected $BoneCount bones and no zero-weight vertices."
}
$comparison = Get-Content -LiteralPath $compareReport -Raw | ConvertFrom-Json
foreach ($sample in @($comparison.frames)) {
    if ([double]$sample.diff.rms -ge $MaxRms) {
        throw "Alembic comparison acceptance failed at frame $($sample.frame): RMS $($sample.diff.rms)"
    }
}
Write-Output "PASS: Dem Bones decomposition, clean Blender rig export, Blender acceptance, and Alembic comparison completed."
Write-Output "Final FBX: $finalOutput"
