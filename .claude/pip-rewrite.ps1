$input_json = $input -join ''
$obj = $input_json | ConvertFrom-Json
$cmd = $obj.tool_input.command
$venv_python = 'C:\Users\vbarr\Documents\11-Codes\Scanner\Scripts\python.exe'
$venv_pip = 'C:\Users\vbarr\Documents\11-Codes\Scanner\Scripts\pip.exe'
if ($cmd -match '\bpip install\b') {
    $newCmd = $cmd -replace '\bpip install\b', "$venv_python $venv_pip install"
    $out = @{
        hookSpecificOutput = @{
            hookEventName = 'PreToolUse'
            updatedInput = @{ command = $newCmd }
        }
    }
    $out | ConvertTo-Json -Depth 5 -Compress
}
