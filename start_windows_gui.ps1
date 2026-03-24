$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$logsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logsDir ("desktop_launcher_" + $timestamp + ".log")

Start-Transcript -Path $logFile -Append

try {
    Write-Host "[Electra-Core] Desktop launcher started"

    if (-not (Test-Path (Join-Path $ProjectRoot "requirements.txt"))) {
        throw "requirements.txt not found in project folder."
    }

    $venvDir = if (Test-Path (Join-Path $ProjectRoot ".venv")) {
        Join-Path $ProjectRoot ".venv"
    } elseif (Test-Path (Join-Path $ProjectRoot "venv")) {
        Join-Path $ProjectRoot "venv"
    } else {
        Join-Path $ProjectRoot ".venv"
    }

    if (-not (Test-Path $venvDir)) {
        Write-Host "[Electra-Core] Creating virtual environment at $venvDir"
        $pyCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { $null }
        if (-not $pyCmd) {
            throw "Python 3 not found."
        }
        Invoke-Expression "$pyCmd -m venv \"$venvDir\""
    }

    $pythonExe = Join-Path $venvDir "Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "Virtual environment python executable not found: $pythonExe"
    }

    Write-Host "[Electra-Core] Upgrading pip"
    & $pythonExe -m pip install --upgrade pip

    Write-Host "[Electra-Core] Installing dependencies"
    & $pythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Filter = "PDF Files (*.pdf)|*.pdf"
    $dialog.Title = "Select voter-roll PDF"
    $dialog.Multiselect = $false

    $pdfPath = ""
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $pdfPath = $dialog.FileName
    } else {
        $fallback = Get-ChildItem -Path $ProjectRoot -Filter *.pdf -File | Select-Object -First 1
        if ($fallback) {
            $pdfPath = $fallback.FullName
        }
    }

    if (-not $pdfPath) {
        throw "No PDF selected or found in project root."
    }

    Write-Host "[Electra-Core] Running extraction for: $pdfPath"
    & $pythonExe (Join-Path $ProjectRoot "main.py") $pdfPath

    [System.Windows.Forms.MessageBox]::Show("Extraction completed successfully.`n`nLog: $logFile", "Electra-Core", "OK", "Information") | Out-Null
}
catch {
    Write-Error $_
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show("Extraction failed.`n`n$($_.Exception.Message)`n`nLog: $logFile", "Electra-Core", "OK", "Error") | Out-Null
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
