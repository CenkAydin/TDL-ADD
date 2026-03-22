$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "venv\Scripts\python.exe"
$features = Join-Path $root "asv2019PS\preprocess_xls-r-300m"
$log = Join-Path $root "pipeline_run.log"

"[$(Get-Date -Format s)] monitor started" | Out-File -FilePath $log -Encoding utf8

while ($true) {
    $train = (Get-ChildItem (Join-Path $features "train\xls-r-300m") -Filter "*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
    $dev = (Get-ChildItem (Join-Path $features "dev\xls-r-300m") -Filter "*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
    $eval = (Get-ChildItem (Join-Path $features "eval\xls-r-300m") -Filter "*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
    "[$(Get-Date -Format s)] train=$train dev=$dev eval=$eval" | Out-File -FilePath $log -Append -Encoding utf8

    if ($train -ge 25380 -and $dev -ge 24844 -and $eval -ge 71237) {
        break
    }
    Start-Sleep -Seconds 60
}

"[$(Get-Date -Format s)] preprocess complete, training starts" | Out-File -FilePath $log -Append -Encoding utf8
& $python "main_train.py" --out_fold "./models/repro" --num_workers 0 --gpu 0 2>&1 | Tee-Object -FilePath $log -Append

"[$(Get-Date -Format s)] training complete, scoring starts" | Out-File -FilePath $log -Append -Encoding utf8
& $python "generate_score_offline.py" --model_folder "./models/repro" --score_dir "./scores/repro" --model_name "TDL_repro" --gpu 0 2>&1 | Tee-Object -FilePath $log -Append

"[$(Get-Date -Format s)] scoring complete, evaluation starts" | Out-File -FilePath $log -Append -Encoding utf8
& $python "eval_ps.py" --score_dir "./scores/repro" 2>&1 | Tee-Object -FilePath $log -Append

"[$(Get-Date -Format s)] pipeline finished" | Out-File -FilePath $log -Append -Encoding utf8