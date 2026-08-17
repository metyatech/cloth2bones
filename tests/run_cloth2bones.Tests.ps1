Describe 'Cloth2Bones public script contract' {
    It 'does not contain a machine-specific absolute default path' {
        $files = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot '..') -File -Recurse -Include '*.ps1', '*.py', '*.md' | Where-Object { $_.FullName -notmatch '\\node_modules\\' -and $_.Name -ne 'run_cloth2bones.Tests.ps1' }
        $content = $files | Get-Content -Raw
        $content | Should -Not -Match '([A-Za-z]:\\Users\\|/Users/)'
    }

    It 'has explicit converter entry points' {
        Test-Path (Join-Path $PSScriptRoot '..\tools\run_cloth2bones.ps1') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\body_motion_poc.py') | Should -BeTrue
    }
}
