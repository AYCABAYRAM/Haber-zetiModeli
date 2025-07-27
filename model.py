from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from preprocess import tokenized_train, tokenized_val, tokenizer

# MODEL YÜKLE
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")  

# EĞİTİM AYARLARI
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,             # Hızlı prototip için 3 epoch
    predict_with_generate=True,     # Özet üretimini aktifleştir
    logging_dir="./logs",           # Log dosyalarının kaydı
    logging_steps=10,
)

# TRAINER
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# EĞİTİMİ BAŞLAT
trainer.train()

#Modeli kaydet
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("\n Eğitim tamamlandı. Model ./saved_model klasörüne kaydedildi.")