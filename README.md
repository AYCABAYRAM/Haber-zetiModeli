# CNN/DailyMail Haber Özetleme Projesi - README
## Proje Tanımı
Bu projede CNN/DailyMail veri seti kullanılarak uzun haber metinlerinden otomatik olarak kısa özetler çıkarabilen bir yapay zeka sistemi geliştirildi. Bu amaç doğrultusunda Google tarafından geliştirilen ve NLP alanında yaygın olarak kullanılan T5 mimarisine sahip 't5-small' modeli tercih edilmiştir. Model eğitim sonrası değerlendirilerek ROUGE-L skoru ile ölçülmüştür. 
Datasete ulaşmak isterseniz, https://github.com/paperswithcode/paperswithcode-data?tab=readme-ov-file linkini kullanabilirsiniz.

<img width="900" height="345" alt="image" src="https://github.com/user-attachments/assets/6504d7b8-63e2-4ad3-91bc-0a7dfef769e7" />
 
## Dosya Açıklamaları
- _load_data.py:_ Veri setini yükler ve ilk 75K satırdan eğitim, 5K doğrulama, 2K test verisi oluşturur.
- _preprocess.py:_ Temizleme ve tokenize işlemlerini yapar. T5 için uygun input-output formatını hazırlar.
- _model.py:_ T5-small modeli ile eğitimi gerçekleştirir. Eğitim sonrası model ve tokenizer ./saved_model klasörüne kaydedilir.
- _review.py:_ Eğitilen model test verisi üzerinde özet üretir, ROUGE-L skorunu hesaplar ve örnek çıktı üretir.
- _open_model.py:_ model.safetensors içeriğini `safetensors` kütüphanesi ile açmak için kullanılan kod.
- _saved_model/:_ Eğitim sonrası kaydedilen model, tokenizer ve config dosyalarının bulunduğu klasör.
- _requirements.txt:_ Projenin çalışması için gerekli Python kütüphanelerini liste.
  
## Model Yapılandırması (default for T5)
- _Model type:_ T5 (Text-to-Text Transformer)
- _Encoder-decoder:_ Aktif
- _Number of Layers:_ 6
- _Attention Head:_ 8
- _D_model:_ 512
- _Dropout:_ 0.1
- _Max length:_ 256
- _Feedforward boyutu:_ 2048
- _Token size:_ 32,128

## Veri Ön İşleme (preprocess.py)
Bu dosya, haber özetleme görevinde kullanılan CNN/DailyMail veri setinin model eğitimine hazır hale getirilmesi için gerekli ön işleme işlemlerini gerçekleştirir. Metin temizleme, 'summarize:' öneki ekleme, tokenizer ile sayısal vektöre dönüştürme ve padding/truncation işlemleri bu dosyada yapılır. T5 mimarisi, görev tanımını anlayabilmesi için her girişin başına 'summarize:' önekini bekler.

***Uygulanan adımlar şunlardır:***
1. __Metin Temizliği:__ Tüm metinler küçük harfe çevrilir, fazla boşluklar kaldırılır, noktalama işaretleri temizlenir.
2. __Giriş Formatı:__ Her haber metninin başına 'summarize:' eklenerek T5 modeline uygun hale getirilir.
3. __Tokenleştirme:__ Giriş metni `max_length=256` ile, özet metni ise `max_length=128` ile tokenize edilir. Uzunluk limiti aşan veriler `truncation=True` ile kesilir, eksik olanlar `padding='max_length'` ile doldurulur.
4. __Dataset Dönüşümü:__ Hugging Face `map()` fonksiyonu ile bu ön işleme fonksiyonu tüm veri kümesine (train/val/test) uygulanır.

***Çıktılar:***
- Tokenized veri kümeleri: tokenized_train, tokenized_val, tokenized_test
- Her bir örnek, input_ids (giriş) ve labels (hedef özet) vektörlerini içerir.
  
 <img width="900" height="487" alt="image" src="https://github.com/user-attachments/assets/28619429-e425-43cd-84b4-b86cb22342e5" />

## Eğitim Parametreleri (model.py)
Bu dosyada, Hugging Face transformers kütüphanesi üzerinden t5-small modeli kullanılarak eğitim süreci gerçekleştirilmiştir. Seq2SeqTrainingArguments ile hiperparametreler tanımlanmış, ardından Seq2SeqTrainer sınıfı ile model eğitilmiş ve ./saved_model klasörüne kaydedilmiştir. Eğitimde hem eğitim hem de doğrulama veri kümeleri kullanılarak her epoch sonunda modelin performansı değerlendirilmiştir.
- _Batch size (train/eval):_ 8
- _Epoch:_ 3
- _Learning rate:_ 2e-5
- _Evaluation:_ her epoch sonunda
- _Weight decay:_ 0.01
- _Logging steps:_ 50
- _predict_with_generate:_ True (beam search ile tahmin üretimi)

 <img width="900" height="602" alt="image" src="https://github.com/user-attachments/assets/9fc96f23-9802-4255-a481-de59df0435f4" />


Bu ekran çıktısında modelin eğitim sürecine ait son epoch’a kadar olan log verileri görülmektedir. Her adımda:
- _loss:_ Eğitim kaybı değerini temsil eder, örneğin 1.303 → modelin hatası azalarak öğrenmeye devam ettiğini gösterir.
- _grad_norm:_ Gradyanların büyüklüğüdür; bu değer çok yükselirse model öğrenmede dengesizlik yaşıyor olabilir, ancak burada stabil seyretmiştir.
- _learning_rate:_ Öğrenme oranı zamanla azalmış ve sonlara doğru 1e-7 mertebesine kadar düşürülmüştür (scheduler tarafından).
- _'epoch':_ Eğitim sürecinin hangi noktasında olunduğunu gösterir (örneğin 2.95 = 2. epoch’un %95’i tamamlanmış).

Eğitim sonunda:
-_train_loss:_ 1.36 ile eğitim tamamlanmış, bu da modelin kabul edilebilir düzeyde öğrendiğini gösterir.
_train_runtime:_ 4626 saniye (yaklaşık 77 dakika)
_train_samples_per_second:_ 48.63 → eğitimin işlem hızı hakkında bilgi verir.
Bu log'lar, modelin eğitim boyunca istikrarlı şekilde öğrenme sürecini tamamladığını doğrular.

## Değerlendirme (review.py)
Model 100 test haberinde __ROUGE-L skoru: 24.79__

Test örneklerinde model özetleri, gerçek özetlere benzer fakat detay eksikliği ve zaman zaman fazla uzama problemi görülebilir.
Model, özet üretiminde anlam bütünlüğünü genelde korumaktadır.
Model, test setinden seçilen 6 farklı haber üzerinde değerlendirildi. Genel olarak özetler, haberin ana fikrini yansıtırken anlam bütünlüğü korunmuştur. Özellikle kısa ve tek odaklı haberlerde başarılı sonuçlar elde edilmiştir. Ancak detaylı içeriklerde bazı önemli bilgiler atlanmıştır. Model genellikle daha kısa ve sade özetler üretmeye eğilimlidir. Ortalama ROUGE-L skoru %24.79 olup, T5-small modelinin hızlı prototipleme için yeterli doğrulukta çıktılar sunduğu görülmüştür.

<img width="900" height="284" alt="image" src="https://github.com/user-attachments/assets/78153c10-b9ab-47a9-9a56-a54a9114d5fd" />

## Gereksinimler (requirements.txt)
transformers==4.35.2
datasets==2.17.1
evaluate==0.4.1
pandas==2.2.2
torch==2.1.2
tqdm==4.66.2

## Notlar
Modelin eğitimi sırasında başlangıçta bazı parametreler düşük tutulmuştur. per_device_train_batch_size değeri ilk olarak 2 idi, ancak eğitim süresini kısaltmak ve donanımı daha verimli kullanmak için 8'e çıkarılmıştır. Benzer şekilde, logging_steps değeri 10 iken, konsol çıktılarının daha seyrek olmasını sağlamak adına 50'ye yükseltilmiştir. Ayrıca, giriş metinleri için kullanılan max_length değeri 512 idi; bellek ve işlem süresi optimizasyonu amacıyla bu değer 256’ya düşürülmüştür. Bu ayarlamalar, modelin daha hızlı ve stabil eğitilmesini sağlamıştır.

## Kullanım Talimatları (Nasıl Çalıştırılır?)
Proje dosyaları sırasıyla aşağıdaki şekilde çalıştırılmalıdır:
1. requirements.txt  
   Ortamda gerekli kütüphanelerin kurulması için aşağıdaki komut çalıştırılır:

       pip install -r requirements.txt

2. load_data.py  
   CNN/DailyMail veri seti `.csv` formatında hazır olmalıdır. Bu dosya çalıştırıldığında veri yüklenir ve ilk 75.000 satır eğitim, 5.000 satır doğrulama, 2.000 satır test olarak ayrılır.

       python load_data.py

3. preprocess.py  
   Eğitim, doğrulama ve test veri setleri üzerinde temizleme ve tokenize işlemleri uygulanır. T5 modeline uygun format hazırlanır.

       python preprocess.py

4. model.py  
   T5-small modeli eğitilir. Eğitilmiş model ve tokenizer `./saved_model/` klasörüne kaydedilir.

       python model.py

5. review.py  
   Kaydedilen model yüklenerek test verisi üzerinde özetler üretilir. ROUGE-L skoru hesaplanır ve örnek karşılaştırmalar konsola yazdırılır.

       python review.py

6. (İsteğe bağlı) open_model.py  
   Eğer eğitilen model dosyasının (`model.safetensors`) içeriği doğrudan incelenmek istenirse, bu dosya ile `safetensors` formatı içindeki katman isimleri görüntülenebilir.

       python open_model.py

