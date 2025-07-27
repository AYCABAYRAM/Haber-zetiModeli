from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import evaluate
import torch
from tqdm import tqdm
from load_data import test_dataset

model = AutoModelForSeq2SeqLM.from_pretrained("./saved_model") 
tokenizer = AutoTokenizer.from_pretrained("./saved_model") 
model.eval() 

print("ROUGE Skoru Hesaplanıyor...")
test_df = pd.read_csv("C:\\gykmodule2\\hw\\hw6_yedek\\test.csv").dropna(subset=["article", "highlights"]) 
test_df = test_df.head(100)
predictions = []

articles = test_df["article"].tolist() 
references = test_df["highlights"].tolist() 

predictions = []
for article in tqdm(articles, desc="Özet Üretiliyor"):
    input_text = "summarize: " + article 
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True) 
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    predictions.append(summary)

rouge = evaluate.load("rouge") 
results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
rouge_l_score = round(results["rougeL"] * 100, 2) 

def summarize_article(article):

    inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=256, truncation=True)
    
    with torch.no_grad(): 
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=64, 
            min_length=10, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True 
        )

    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip() 
    return generated_summary

    """Verilen test indeksindeki haberi özetler ve karşılaştırmalı çıktı verir."""
def summarize_and_compare(index):
    article = test_df["article"][index]
    reference = test_df["highlights"][index]
    generated = summarize_article(article)

    print(f"\nÖRNEK {index + 1}")
    print("🔹 ORİJİNAL HABER (İlk 300 karakter):")
    print(article[:300] + "...\n")
    
    print("GERÇEK ÖZET:")
    print(reference + "\n")
    
    print("MODEL TARAFINDAN ÜRETİLEN ÖZET:")
    print(generated + "\n")
    
    print(f"Uzunluk karşılaştırması: Model ({len(generated.split())} kelime) | Gerçek ({len(reference.split())} kelime)")

test_indices = [0, 7, 15, 23, 45, 66]

print("CNN NEWS SUMMARIZATION KARŞILAŞTIRMALI ÖRNEKLER")

for idx in test_indices:
    summarize_and_compare(idx)

print("\nToplam", len(test_indices), "örnek başarıyla işlendi.")
