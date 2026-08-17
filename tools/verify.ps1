[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
python -m ruff check (Join-Path $repoRoot 'cloth2bones') (Join-Path $repoRoot 'tools') (Join-Path $repoRoot 'tests')
python -m pyright (Join-Path $repoRoot 'cloth2bones')
python -m pytest -q
Invoke-ScriptAnalyzer -Path (Join-Path $repoRoot 'tools\run_cloth2bones.ps1') -Severity Warning,Error
Invoke-ScriptAnalyzer -Path (Join-Path $repoRoot 'tools\verify.ps1') -Severity Warning,Error
Invoke-Pester -Path (Join-Path $repoRoot 'tests') -Output Detailed
if (Get-Command npx -ErrorAction SilentlyContinue) {
    Push-Location $repoRoot
    try {
        npx --no-install markdownlint-cli2 '*.md' 'docs/*.md' 'examples/*.md'
    }
    finally {
        Pop-Location
    }
}
Write-Output 'PASS: static analysis, PowerShell analysis, tests, and Markdown checks completed.'
