## 1. PREPROCESS
Proje çalışmadan önce preprocess.py dosyasını çalıştırmak gerekir. Bu dosya ASVspoof2019PS veri setini preprocess eder. 
Girdi olarak ham ses dosyası alır .wav formatında. 
Örneklemeyi zorunlu olarak 16000 Hz'a çevirir. Mesela 4 saniye ses varsa 64000 adet örneklem olur. 
Wav2Vec2-XLS-R her 20 milisaniyelik ses için 1024 boyutlu bir vektör çıkarır. 
.pt formatında kaydeder. Örnek olarak 4 saniyeli ses için (200,1024) boyutunda bir matris çıkarır.

## 2. DATASET
Eğitim başladığında veriler kaydedilmiş olan .pt dosyalarından okunur ve modele sokulacak standart paketler haline getirilir.
İlgili kod dataset.py ve main.py ASVspoof2019PS class'ı
1024 tane sütun vardı x adet satır var. Bu satırlar sesin 20 milisaniyelik parçalarını temsil ediyor. 
1 frame = 1 satır
Bu aşamada her ses dosyası için CNN sabit giriş boyutu istiyor (kimi ses 2 saniye kimisi 10 saniye).
Hedef boyut 1050 frame belirlenmiş default.
Eğer dosya uzunsa rastgele bir yerinden 1050 framelik yer kesilir. Dosya kısaysa zero padding ile 1050 frame haline getirilir.
Ayrıca ek olarak batch işlemi yapılır örnek olarak 32 dosya üst üste koyulur. Sonuç olarak dataset.py içeriğinden (32,1050,1024) boyutunda bir matris çıkarılır. Batch, zaman, feature(özellik)  sırasıyla.
Pytorch Conv1d katmanları özellik zaman olarak ister. O yüzden transpose işlemi yapılır. Sonuç olarak (32,1024,1050) boyutunda bir matris çıkarılır.
Veri modele (Batch, 1024, 1050) şeklinde sokulur.

## 3. MODEL

### 3.1 Embedding Module
Bu kısım model.py içinde EmbeddingModule class'ı
1024 çok büyük olduğu için Conv1d ile 1024 kanalı önce 512 sonra 32'ye düşür.
L2 normalizasyon yap.
Çıktı(embedding) (Batch, 32, 1050) şeklindedir.
Bu çıktı sonraki aşamada navigasyon haritası olarak kullanılır.

### 3.2 Base Module
Bu kısım model.py içinde BaseModule class'ı ve AttentionConv1d class'ı
Burada attention mekanizması bulunuyor. 
Girdi olarak orijinal veri (Batch, 1024, 1050) ve navigasyon haritası (Batch, 32, 1050) alır.
Çalışma mantığı şu şekilde:
1. Benzerlik hesabı: 32 boyutlu embedding'i kullanarak her frame'in (t) sağındaki (t+1) ve solundaki (t-1) komşusuyla arasındaki açıyı yani Cosine Similarity hesaplar. Sonuç 1'e yakınsa biz aynıyız (ikimiz de real veya fake), sonuç 0 veya negatifse biz farklıyız (biri fake biri real).
2. Attention maskesi: Bu benzerlik skorları bir ağırlık matrisine dönüştürülür.
3. Çarpma işlemi: Orijinal 1024'lük veri bu benzerlik maskesi ile çarpılır. Eğer bir frame fake yanındaki real ise aradaki benzerlik düşük çıkar ve maske o komşudan gelen bilgisi sıfırlar veya azaltır. Böylece sahte ve gerçek arasındaki sınır belli olur.
Çıktı (Batch, 1024, 1050) şeklindedir. Boyut değişmez 

### 3.3 Classifier Module
Kodda model.py ClassifierModule class'ı
Conv1d ile 1024 kanalı 2 kanala düşürür. Kanal 0 fake olma ihtimali Kanal 1 real olma ihtimali. Çıktı (Batch, 2, 1050) şeklindedir. 
Flatten ile veriyi düzleştirir (Batch, 2100) şeklindedir. 
Linear(FC) ile 2100 boyutlu veriyi 132 boyutlu veriye düşürür. 
Sigmoid ile 0-1 arası değerler elde edilir. 
Nihai çıktı (Batch, 132) boyutlu bir vektördür. 
Neden 132? Çünkü ASVspoof2019PS veri setinde etiketler her frame için değil belirli bloklar için verilir. 1050 framelik ses çözünürlük düşürülerek 132 parçalık bir karara bağlanır.

## 4. TRAIN
Model tahmin yaptı 132 sayı üretti doğru mu?
Kodda main_train.py
Toplam kayıp loss formülü: Loss = BCE_Loss + (lambda * Embedding_Loss)
Binary Cross Entropy. Modelin ürettiği 132 skor ile gerçek etiket (Ground Truth) karşılaştırılır.
Embedding_loss. Modelin 32 boyutlu ürettiği o haritayı denetler.