from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import evaluate
import torch
from tqdm import tqdm

# Model ve tokenizer'ı yükle (senin eğittiğin)
model = AutoModelForSeq2SeqLM.from_pretrained("./saved_model") #Daha önce eğittiğim ve ./saved_model klasörüne kaydettiğin T5-small modeli yeniden yükleniyor. AutoModelForSeq2SeqLM sınıfı seq2seq görevleri için (örneğin özetleme) uygun model yapısını çağırır.
tokenizer = AutoTokenizer.from_pretrained("./saved_model") #tokenizer da yine aynı klasörden yükleniyor.

# Test setini oku
test_df = pd.read_csv("C:\\gykmodule2\\hw\\hw6_yedek\\test.csv").dropna(subset=["article", "highlights"]) #dropna ifadesi, özet veya haber kısmı boş olan satırları siler.
test_df = test_df.head(100)  # hızlı test için ilk 100 örnek alınabilir

articles = test_df["article"].tolist() #articles: Modelin özetleyeceği giriş metinleri (haberler).
references = test_df["highlights"].tolist() #references: Gerçek özetler (etiketler) – değerlendirme için kullanılacak.

# Model özetleri üret
predictions = []
for article in tqdm(articles, desc="Özet Üretiliyor"):
    input_text = "summarize: " + article #summarize t5 in görev tanımı için eklenir
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True) #model.generate(...): Özet üretir.
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True) #decode(...): Sayıları tekrar metne çevirir.
    predictions.append(summary)

# ROUGE hesapla
rouge = evaluate.load("rouge") #ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Otomatik özetleme kalitesini ölçen metrik.
results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
rouge_l_score = round(results["rougeL"] * 100, 2) #rougeL: Cümledeki en uzun ortak alt dizi (Longest Common Subsequence) üzerinden ölçüm yapar. Skor yüzdeye çevrilip yuvarlanır.

# Sonuçları yazdır
print("\nROUGE-L Skoru:", rouge_l_score)
print("\nÖrnek Çıktı:")
for i in range(5):
    print(f"\n--- Örnek {i+1} ---")
    print("Haber:", articles[i][:200], "...") #haber metninin ilk 200 karakteri
    print("Gerçek Özet:", references[i]) #gerçek özet
    print("Model Özeti :", predictions[i]) #modelin özeti



