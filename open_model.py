from safetensors.torch import load_file

# Model dosyasının tam yolu — bunu kendi dosya konumuna göre düzelt
tensors = load_file("C:/gykmodule2/saved_model/model.safetensors")

# Tüm katman ağırlıklarının isimlerini yazdır
print(tensors.keys())

