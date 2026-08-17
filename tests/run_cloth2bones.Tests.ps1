Describe 'Cloth2Bones public script contract' {
    It 'does not contain a machine-specific absolute default path' {
        $files = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot '..') -File -Recurse -Include '*.ps1', '*.py', '*.md' | Where-Object { $_.FullName -notmatch '\\node_modules\\' -and $_.Name -ne 'run_cloth2bones.Tests.ps1' }
        $content = $files | Get-Content -Raw
        $content | Should -Not -Match '([A-Za-z]:\\Users\\|/Users/)'
    }

    It 'has explicit converter entry points' {
        Test-Path (Join-Path $PSScriptRoot '..\tools\run_cloth2bones.ps1') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\body_motion_poc.py') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\run_body_local_poc.ps1') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\body_local_poc.py') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\run_cross_sequence_poc.ps1') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\cross_sequence_poc.py') | Should -BeTrue
        Test-Path (Join-Path $PSScriptRoot '..\tools\audit_cross_sequence_dataset.py') | Should -BeTrue
    }

    It 'documents the same-rig body-local safety contract' {
        $content = Get-Content -Raw (Join-Path $PSScriptRoot '..\tools\apply_body_driven_poses.py')
        $content | Should -Match 'validate_rig_pose_contract'
        $content | Should -Match 'export poses from the same clean rig'
    }
}
