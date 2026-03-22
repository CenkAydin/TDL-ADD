# TDL-ADD Uctan Uca Calistirma (Windows)

Bu dosya yeni bilgisayarda sirayla komutlari kopyala-yapistir calistirarak ilerlemen icin hazirlandi.

## 0) Gerekenler

- NVIDIA GPU ve guncel surucu
- En az 250 GB bos disk (onerilir)
- Python 3.11.x
- Git

## 1) Projeyi indir

Asagidaki komutta `<GITHUB_REPO_URL>` yerine kendi repo URL'ni yaz:

```powershell
cd C:\Users\Public; git clone <GITHUB_REPO_URL>; cd TDL-ADD
```

## 2) Sanal ortam ve paket kurulumu

```powershell
cd C:\Users\Public\TDL-ADD; py -3.11 -m venv venv; .\venv\Scripts\Activate.ps1; python -m pip install --upgrade pip
```

CUDA 12.1 tekeri ile PyTorch kurulumu:

```powershell
.\venv\Scripts\Activate.ps1; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Proje bagimliliklari:

```powershell
.\venv\Scripts\Activate.ps1; pip install numpy scipy scikit-learn librosa soundfile tqdm transformers safetensors pytorch-model-summary
```

## 3) Veri setini yerlestir

Zenodo 5766198 icerigini ac ve su yapinin oldugunu kontrol et:

- `asv2019PS/database/train/con_wav`
- `asv2019PS/database/dev/con_wav`
- `asv2019PS/database/eval/con_wav`
- `asv2019PS/database/protocols`
- `asv2019PS/database/segment_labels`

Not: repodaki `label` klasoru oldugu gibi kalmali.

## 4) Hızlı kontrol

CUDA gorunuyor mu:

```powershell
.\venv\Scripts\Activate.ps1; python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
```

Scriptler derleniyor mu:

```powershell
.\venv\Scripts\Activate.ps1; python -m py_compile preprocess.py dataset.py main_train.py generate_score_offline.py eval_ps.py
```

## 5) Feature extraction (uzun surebilir)

```powershell
.\venv\Scripts\Activate.ps1; python preprocess.py
```

Ilerleme sayaci:

```powershell
.\venv\Scripts\Activate.ps1; $b='C:\Users\Public\TDL-ADD\asv2019PS\preprocess_xls-r-300m'; 'train=' + (Get-ChildItem "$b\train\xls-r-300m" -Filter '*.pt' -ErrorAction SilentlyContinue | Measure-Object).Count; 'dev=' + (Get-ChildItem "$b\dev\xls-r-300m" -Filter '*.pt' -ErrorAction SilentlyContinue | Measure-Object).Count; 'eval=' + (Get-ChildItem "$b\eval\xls-r-300m" -Filter '*.pt' -ErrorAction SilentlyContinue | Measure-Object).Count
```

Beklenen tamam sayilari:

- train: 25380
- dev: 24844
- eval: 71237

## 6) Egitim

```powershell
.\venv\Scripts\Activate.ps1; python main_train.py --out_fold ./models/repro --num_workers 0 --gpu 0
```

Model dosyasi olusmali:

- `models/repro/anti-spoofing_feat_model.pt`

## 7) Skor uretme

```powershell
.\venv\Scripts\Activate.ps1; python generate_score_offline.py --model_folder ./models/repro --score_dir ./scores/repro --model_name TDL_repro --gpu 0
```

Skor dosyalari olusmali:

- `scores/repro/final_label.npy`
- `scores/repro/final_pred.npy`

## 8) Metrik hesaplama

```powershell
.\venv\Scripts\Activate.ps1; python eval_ps.py --score_dir ./scores/repro
```

Ekranda EER, Precision, Recall, F1 gorursun.

## 9) Sonucu makale ile karsilastirma

Gorseldeki 19PS hedef EER degeri: 7.04.

Karsilastirma:

- Senin EER
- 7.04
- Fark = Senin EER - 7.04

## 10) Disk yonetimi

Bos alan kontrolu:

```powershell
Get-PSDrive -Name C | Select-Object Name,@{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}},@{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}
```

Preprocess feature klasorleri cok yer kaplar:

- `asv2019PS/preprocess_xls-r-300m/train/xls-r-300m`
- `asv2019PS/preprocess_xls-r-300m/dev/xls-r-300m`
- `asv2019PS/preprocess_xls-r-300m/eval/xls-r-300m`

Calisma bitince yer acmak icin silebilirsin.
