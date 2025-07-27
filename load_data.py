import pandas as pd
from datasets import Dataset
import os

# CSV dosyalarını oku
train_df = pd.read_csv("C:\\gykmodule2\\hw\\hw6_haber\\train.csv")
val_df = pd.read_csv("C:\\gykmodule2\\hw\\hw6_haber\\validation.csv")
test_df = pd.read_csv("C:\\gykmodule2\\hw\\hw6_haber\\test.csv")

# Huggingface Dataset'e çevir
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# İlk 5 satırı yazdır
print("\nTrain:")
print(train_df.head())

print("\nValidation:")
print(val_df.head())

print("\nTest:")
print(test_df.head())
