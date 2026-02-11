# 🚀 TensorFlow - Ściągawka dla Studentów

## Szybki dostęp do najważniejszych informacji

### 📖 Dokumentacja - Kolejność czytania

1. **START HERE** 👉 [STUDENT_GUIDE_PL.md](STUDENT_GUIDE_PL.md)
2. **Podstawy** → [README_PL.md](README_PL.md)
3. **Robot G1** → [UNITREE_G1_GUIDE_PL.md](UNITREE_G1_GUIDE_PL.md)
4. **Zastosowania** → [ROBOTICS_APPLICATIONS_PL.md](ROBOTICS_APPLICATIONS_PL.md)

### 💻 Przykłady kodu

| Co chcesz zrobić | Gdzie szukać |
|------------------|--------------|
| Rozpoznawać obiekty | [label_image/](tensorflow/examples/label_image/) → [README_PL.md](tensorflow/examples/label_image/README_PL.md) |
| Sterowanie głosem | [speech_commands/](tensorflow/examples/speech_commands/) → [README_PL.md](tensorflow/examples/speech_commands/README_PL.md) |
| Wszystkie przykłady | [examples/](tensorflow/examples/) → [README_PL.md](tensorflow/examples/README_PL.md) |

---

## ⚡ Instalacja - Krok po kroku

```bash
# 1. Zainstaluj TensorFlow
pip install tensorflow

# 2. Zainstaluj dodatkowe biblioteki
pip install numpy pillow matplotlib

# 3. (Opcjonalnie) Dla GPU
pip install tensorflow-gpu

# 4. Sprawdź instalację
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

## 🎯 Pierwszy projekt w 5 minut

### Rozpoznawanie obrazów

```bash
# 1. Przejdź do katalogu
cd tensorflow/examples/label_image

# 2. Pobierz model
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -xz -C data/

# 3. Uruchom
python3 label_image.py

# 4. Testuj na swoim zdjęciu
python3 label_image.py --image=twoje_zdjecie.jpg
```

**Gotowe!** Robot rozpoznaje co jest na zdjęciu! 🎉

---

## 📝 Najczęstsze polecenia

### Trening modelu mowy

```bash
# Z domyślnymi danymi (angielski)
python train.py

# Z polskimi komendami
python train.py \
  --data_dir=moje_nagrania \
  --wanted_words=idz,stop,lewo,prawo
```

### Testowanie modelu

```bash
# Pojedynczy plik
python label_wav.py --wav=test.wav

# Cały katalog
python label_wav_dir.py --wav_dir=testy/
```

### Konwersja do TensorFlow Lite

```bash
# Zamrożenie modelu
python freeze.py \
  --start_checkpoint=checkpoint.ckpt \
  --output_file=model.pb

# Konwersja do .tflite
python convert_to_tflite.py --model=model.pb
```

---

## 🔧 Parametry które warto znać

### label_image.py

```bash
--image          # Ścieżka do obrazu
--graph          # Model (.pb file)
--labels         # Plik z nazwami kategorii
--input_height   # Wysokość obrazu (299 dla Inception)
--input_width    # Szerokość obrazu (299 dla Inception)
```

### train.py (speech_commands)

```bash
--data_dir                  # Katalog z danymi
--wanted_words              # Słowa do rozpoznawania (oddzielone przecinkami)
--how_many_training_steps   # Liczba kroków treningu
--learning_rate             # Szybkość uczenia
--batch_size                # Rozmiar batcha
```

---

## 🐛 Rozwiązywanie problemów

### Problem: ModuleNotFoundError

```bash
# Zainstaluj brakującą bibliotekę
pip install nazwa_biblioteki

# Przykład:
pip install numpy tensorflow pillow
```

### Problem: Model działa wolno

**Rozwiązanie:**
1. Użyj TensorFlow Lite (konwersja powyżej)
2. Zmniejsz rozdzielczość obrazu
3. Użyj mniejszego modelu (MobileNet zamiast Inception)

### Problem: Niska dokładność

**Rozwiązanie:**
1. Zbierz więcej danych (najważniejsze!)
2. Trenuj dłużej (więcej kroków)
3. Użyj data augmentation
4. Wypróbuj lepszy model

### Problem: CUDA errors (GPU)

**Rozwiązanie:**
```bash
# Użyj CPU version
pip uninstall tensorflow-gpu
pip install tensorflow-cpu
```

---

## 📊 Typowe wartości parametrów

### Dla obrazów

```python
input_height = 299      # Inception V3
input_width = 299
input_mean = 0
input_std = 255

# LUB dla MobileNet
input_height = 224
input_width = 224
```

### Dla audio

```python
sample_rate = 16000           # 16 kHz
clip_duration_ms = 1000       # 1 sekunda
window_size_ms = 30           # Okno analizy
window_stride_ms = 10         # Przesunięcie okna
```

### Trening

```python
learning_rate = 0.001         # Początkowa szybkość
batch_size = 100              # Dla większości zadań
training_steps = 15000        # Dobry start
```

---

## 🎨 Struktura projektu robotycznego

```
moj_projekt/
├── data/                    # Dane treningowe
│   ├── images/             # Zdjęcia
│   └── audio/              # Nagrania
├── models/                  # Modele TensorFlow
│   ├── vision.pb
│   └── speech.pb
├── src/                     # Kod źródłowy
│   ├── robot_control.py
│   ├── vision.py
│   └── speech.py
├── tests/                   # Testy
└── docs/                    # Dokumentacja
```

---

## 📞 Gdzie szukać pomocy

1. **Dokumentacja w tym repo (PL):**
   - [STUDENT_GUIDE_PL.md](STUDENT_GUIDE_PL.md) - Przewodnik główny

2. **Oficjalna dokumentacja:**
   - https://www.tensorflow.org/tutorials

3. **Forum i społeczność:**
   - https://discuss.tensorflow.org/
   - Stack Overflow (tag: tensorflow)

4. **GitHub Issues:**
   - https://github.com/tensorflow/tensorflow/issues

---

## ✅ Checklist przed rozpoczęciem

- [ ] TensorFlow zainstalowany i działa
- [ ] Przeczytany STUDENT_GUIDE_PL.md
- [ ] Uruchomiony pierwszy przykład
- [ ] Zrozumiane podstawy (tensor, model, training)

## ✅ Checklist przed deploymentem na robota

- [ ] Model przetestowany offline
- [ ] Skonwertowany do TensorFlow Lite
- [ ] Accuracy > 90% (lub akceptowalne dla zadania)
- [ ] Czas inferencji < 100ms (dla real-time)

---

## 🎯 Kluczowe koncepty - Minimalne wymagania

### Musisz rozumieć:
- ✅ **Tensor** - Wielowymiarowa tablica liczb
- ✅ **Model** - Wytrenowana sieć neuronowa
- ✅ **Preprocessing** - Przygotowanie danych
- ✅ **Inference** - Używanie modelu do predykcji
- ✅ **Accuracy** - Miara dokładności modelu

### Dobrze by było wiedzieć:
- 📚 Jak działa backpropagation
- 📚 Różnice między architekturami (CNN, RNN)
- 📚 Overfitting i jak go unikać
- 📚 Transfer learning

### Nice to have:
- 🎓 Matematyka uczenia maszynowego
- 🎓 Optymalizacja hyperparametrów
- 🎓 Własne architektury sieci
- 🎓 Research i publikacje

---

## 💡 Szybkie wskazówki

### DO ✅
- Zaczynaj od prostych przykładów
- Używaj pretrained models
- Dokumentuj swój kod
- Testuj często
- Pytaj gdy nie rozumiesz

### NIE RÓB ❌
- Nie trenuj od zera bez powodu
- Nie ignoruj warnings
- Nie pomijaj preprocessing
- Nie zaniedbuj testów
- Nie bój się błędów

---

**Powodzenia!** 🚀

*Masz pytania? Zobacz [STUDENT_GUIDE_PL.md](STUDENT_GUIDE_PL.md)*
