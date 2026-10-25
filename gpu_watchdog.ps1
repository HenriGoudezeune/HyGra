# Logs GPU temperature every 30 s; stops training if the GPU stays >= 87 C for a minute.
param([int]$StopTemp = 87, [int]$WarnTemp = 80)
$log = Join-Path $PSScriptRoot "logs\gpu_temp.log"
$hot = 0
while ($true) {
    $t = 0
    if ([int]::TryParse((nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits).Trim(), [ref]$t)) {
        $line = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $t
        if ($t -ge $WarnTemp) { $line += " WARN" }
        if ($t -ge $StopTemp) { $hot++ } else { $hot = 0 }
        if ($hot -ge 2) {
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                Where-Object { $_.CommandLine -match 'train_(tokenizer|generator)' } |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
            $line += " STOPPED-TRAINING"
            $hot = 0
        }
        for ($i = 0; $i -lt 5; $i++) {
            try { Add-Content -Path $log -Value $line -Encoding ascii -ErrorAction Stop; break }
            catch { Start-Sleep -Milliseconds 200 }
        }
    }
    Start-Sleep -Seconds 30
}
