import re
from transformers import AutoTokenizer
from load_data import train_dataset, val_dataset, test_dataset

# Tokenizer'ı yükle
model_name = "t5-small"  # veya "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Temizlik fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Huggingface Dataset'te map ile kullanılacak preprocess fonksiyonu
def preprocess_function(examples):
    cleaned_articles = [clean_text(article) for article in examples["article"]]
    cleaned_summaries = [clean_text(summary) for summary in examples["highlights"]]

    # T5 için "summarize: " prefix'i veriyoruz
    inputs = ["summarize: " + article for article in cleaned_articles]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length") #max_length=512 → input için sınır koyduk, truncation=True → uzun olanları kesiyoruz, padding="max_length" → kısa olanlara boşluk doldurduk

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(cleaned_summaries, max_length=128, truncation=True, padding="max_length") #max_length=128 → özet (label) için sınır koyduk, truncation=True → uzun olanları kesiyoruz, padding="max_length" → kısa olanlara boşluk doldurduk

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Dataset'leri tokenize et
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# (Opsiyonel) örnek satır yazdır
print("\nÖrnek tokenized_train[0]:")
print(tokenized_train[0])
