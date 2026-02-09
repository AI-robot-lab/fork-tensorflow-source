# Przykłady TensorFlow C++ - Przewodnik dla Studentów

## Wprowadzenie

Ten katalog zawiera przykłady wykorzystania TensorFlow API w języku C++. Jeśli szukasz innych zasobów, sprawdź:

* **Przykłady Python TensorFlow** - zobacz [samouczki na tensorflow.org](https://tensorflow.org/tutorials)
* **Przykłady Keras** - zobacz [keras.io/examples](https://keras.io/examples/)
* **Przykłady TensorFlow Lite** - zobacz [repozytorium tensorflow/examples](https://github.com/tensorflow/examples/tree/master/lite)
* **Notatniki kursu Udacity** - zobacz [ten katalog](https://github.com/tensorflow/examples/tree/master/courses)

## O tych przykładach

⚠️ **Ważne informacje:**

* API C++ TensorFlow można łatwo budować tylko w ramach systemu budowania `bazel` TensorFlow. Jeśli potrzebujesz samodzielnej kompilacji, zobacz [C API](https://www.tensorflow.org/install/lang_c).
* Ten katalog nie jest aktywnie utrzymywany - przykłady mogą być przestarzałe.

**Dla większości projektów zalecamy używanie API Python**, które jest bardziej kompletne i łatwiejsze w użyciu.

## Dostępne przykłady

### 1. Rozpoznawanie obrazów (label_image)

📁 **Katalog:** `label_image/`

**Co robi:**
- Klasyfikuje obiekty na obrazach
- Używa modelu Inception V3
- Rozpoznaje 1000 kategorii obiektów

**Dlaczego jest ważny dla robotyki:**
- Robot może rozpoznawać przedmioty w swoim otoczeniu
- Podstawa dla systemów widzenia komputerowego
- Przydatny do nawigacji i manipulacji obiektami

**Dokumentacja:**
- [README_PL.md](label_image/README_PL.md) - Szczegółowy przewodnik po polsku
- [README.md](label_image/README.md) - Oryginalna dokumentacja

**Jak uruchomić:**
```bash
# Python
python3 label_image.py --image=moj_obraz.jpg

# C++ (wymaga bazel)
bazel run label_image -- --image=moj_obraz.jpg
```

**Zastosowanie dla Unitree G1:**
- Rozpoznawanie obiektów do manipulacji
- Identyfikacja przeszkód
- Rozpoznawanie osób

### 2. Rozpoznawanie poleceń głosowych (speech_commands)

📁 **Katalog:** `speech_commands/`

**Co robi:**
- Rozpoznaje krótkie komendy głosowe
- Można trenować na własnych słowach (również po polsku!)
- Działa w czasie rzeczywistym

**Dlaczego jest ważny dla robotyki:**
- Robot może reagować na polecenia głosowe
- Sterowanie bez użycia rąk
- Naturalna interakcja człowiek-robot

**Dokumentacja:**
- [README_PL.md](speech_commands/README_PL.md) - Kompletny przewodnik po polsku
- [README.md](speech_commands/README.md) - Oryginalna dokumentacja

**Jak uruchomić:**
```bash
# Trening modelu
python3 train.py --wanted_words=idz,stop,lewo,prawo

# Testowanie
python3 label_wav.py --wav=test.wav
```

**Zastosowanie dla Unitree G1:**
- Sterowanie ruchem robota głosem
- Polecenia manipulatora
- Interakcja z użytkownikiem

### 3. Tworzenie własnych operacji (adding_an_op)

📁 **Katalog:** `adding_an_op/`

**Co robi:**
- Pokazuje jak tworzyć własne operacje TensorFlow
- Przykłady w C++ i Python
- Integracja z GPU (CUDA)

**Dlaczego jest ważny:**
- Optymalizacja wydajności dla specyficznych zadań
- Implementacja niestandardowych algorytmów
- Rozszerzanie możliwości TensorFlow

**Dokumentacja:**
- [README.md](adding_an_op/README.md) - Przewodnik tworzenia operacji

**Dla kogo:**
- Zaawansowani użytkownicy
- Optymalizacja wydajności krytycznych części kodu

### 4. Własne operacje - dokumentacja (custom_ops_doc)

📁 **Katalog:** `custom_ops_doc/`

**Co zawiera:**
- Szczegółowe przykłady tworzenia własnych operacji
- Różne poziomy złożoności (multiplex_1, multiplex_2, etc.)
- Integracja z gradientami (backward pass)

**Podkatalogi:**
- `multiplex_1/` - Podstawowa operacja
- `multiplex_2/` - Z gradienty
- `multiplex_3/` - Z shape inference
- `multiplex_4/` - Pełna implementacja
- `simple_hash_table/` - Implementacja hash table
- `sleep/` - Operacja asynchroniczna

### 5. Inne przykłady

#### Przetwarzanie audio (wav_to_spectrogram)
- Konwersja plików WAV do spektrogramów
- Użyteczne dla dalszej analizy audio

#### Ponowne trenowanie modeli (image_retraining)
- Transfer learning dla własnych danych
- Dostosowanie pretrained modeli

#### Android
- Integracja TensorFlow na urządzeniach Android
- TensorFlow Lite dla aplikacji mobilnych

## Struktura katalogów

```
tensorflow/examples/
├── README_PL.md                    # Ten plik
├── README.md                       # Oryginalna dokumentacja
│
├── label_image/                    # Rozpoznawanie obrazów ⭐
│   ├── README_PL.md
│   ├── label_image.py             # Wersja Python (z komentarzami PL)
│   └── main.cc                     # Wersja C++
│
├── speech_commands/                # Rozpoznawanie mowy ⭐
│   ├── README_PL.md
│   ├── train.py                    # Trenowanie modelu (z komentarzami PL)
│   ├── label_wav.py               # Testowanie
│   └── freeze.py                   # Tworzenie wersji produkcyjnej
│
├── adding_an_op/                   # Własne operacje
│   └── README.md
│
├── custom_ops_doc/                 # Dokumentacja własnych operacji
│   ├── multiplex_1/
│   ├── multiplex_2/
│   └── ...
│
├── wav_to_spectrogram/            # Przetwarzanie audio
├── image_retraining/              # Transfer learning
├── android/                        # Przykłady Android
└── udacity/                        # Materiały kursu Udacity
```

## Rekomendowane ścieżki nauki

### Dla początkujących (2-4 tygodnie)

```
Tydzień 1-2: Podstawy
├─ Zapoznanie z TensorFlow Python API
├─ Przejście przez tutorial label_image
│  └─ Uruchomienie na własnych obrazach
└─ Zrozumienie kodu z komentarzami

Tydzień 3-4: Pierwszy projekt
├─ Trening modelu speech_commands
├─ Zbieranie własnych danych
└─ Testowanie na prawdziwych danych
```

### Dla średnio zaawansowanych (4-8 tygodni)

```
Tydzień 1-3: Vision
├─ Transfer learning z image_retraining
├─ Fine-tuning na własnych danych
└─ Integracja z robotem (symulacja)

Tydzień 4-6: Speech
├─ Trening na polskich komendach
├─ Optymalizacja dla czasu rzeczywistego
└─ Integracja z mikrofonami robota

Tydzień 7-8: Integracja
├─ Łączenie vision + speech
├─ Deployment na robocie
└─ End-to-end testing
```

### Dla zaawansowanych (8+ tygodni)

```
├─ Własne architektury sieci
├─ Implementacja custom operations
├─ Multi-task learning
├─ Reinforcement learning
└─ Research & publikacje
```

## Praca z robotem Unitree G1 EDU-U6

### Typowy workflow projektu

```
1. DEVELOPMENT (na komputerze)
   ├─ Zbierz dane z robota
   ├─ Wytrenuj model
   └─ Przetestuj offline

2. OPTIMIZATION
   ├─ Konwertuj do TensorFlow Lite
   ├─ Kwantyzacja
   └─ Benchmark wydajności

3. DEPLOYMENT (na robocie)
   ├─ Transfer modelu
   ├─ Integracja z SDK robota
   └─ Testowanie w realnym środowisku

4. MONITORING & IMPROVEMENT
   ├─ Zbieraj nowe dane
   ├─ Retrain model
   └─ Deploy aktualizacji
```

### Przykładowa integracja

```python
# Pseudo-kod integracji TensorFlow z robotem
import unitree_sdk
import tensorflow as tf

# Inicjalizacja robota
robot = unitree_sdk.G1Robot()

# Załaduj modele TensorFlow
vision_model = tf.lite.Interpreter("object_detector.tflite")
speech_model = tf.lite.Interpreter("speech_commands.tflite")

# Główna pętla robota
while robot.is_active():
    # Percepcja
    image = robot.camera.capture()
    audio = robot.microphone.record()
    
    # Analiza TensorFlow
    objects = vision_model.detect(image)
    command = speech_model.recognize(audio)
    
    # Reakcja robota
    if command == "podnieś" and "kubek" in objects:
        robot.pick_up_object("kubek")
    
    elif command == "idź":
        robot.move_forward()
    
    # ...etc
```

## Narzędzia pomocnicze

### Do developmentu
- **Visual Studio Code** - Z rozszerzeniem Python
- **PyCharm** - IDE dla Pythona
- **Jupyter Notebooks** - Interaktywne eksperymenty
- **Google Colab** - Darmowe GPU do treningu

### Do wizualizacji
- **TensorBoard** - Monitoring treningu
- **Matplotlib** - Wykresy i wizualizacje
- **OpenCV** - Przetwarzanie obrazów

### Do zarządzania danymi
- **LabelImg** - Annotacja obrazów
- **Audacity** - Edycja audio
- **Roboflow** - Zarządzanie datasetami

## Troubleshooting

### Problem: Nie mogę skompilować przykładów C++

**Rozwiązanie:**
- Użyj wersji Python (zazwyczaj wystarczająca)
- Jeśli musisz C++, zobacz [C API documentation](https://www.tensorflow.org/install/lang_c)
- Rozważ TensorFlow Lite C++ API (lżejsze)

### Problem: Przykłady są przestarzałe

**Rozwiązanie:**
- Sprawdź nowsze tutoriale na [tensorflow.org/tutorials](https://tensorflow.org/tutorials)
- Zobacz oficjalne [TensorFlow examples repository](https://github.com/tensorflow/examples)
- Korzystaj z dokumentacji Python API (lepiej utrzymana)

### Problem: Chcę więcej przykładów

**Zasoby:**
- [TensorFlow Hub](https://tfhub.dev/) - Gotowe modele
- [TensorFlow Model Garden](https://github.com/tensorflow/models) - Oficjalne implementacje
- [Papers with Code](https://paperswithcode.com/) - Implementacje z paper'ów

## Dalsze kroki

### Po opanowaniu podstaw:

1. **Eksperymentuj** 
   - Modyfikuj parametry
   - Testuj różne architektury
   - Próbuj własnych pomysłów

2. **Buduj projekty**
   - Zacznij od prostych
   - Stopniowo zwiększaj złożoność
   - Dokumentuj swój kod

3. **Dziel się wiedzą**
   - Pomóż innym studentom
   - Publikuj swoje projekty
   - Contribute do open source

4. **Naucz się więcej**
   - Kursy online
   - Czytaj paper'y
   - Uczestnictwo w konferencjach

## Dodatkowe zasoby edukacyjne

### Przewodniki w tym repozytorium (PL)
- 📘 [README_PL.md](../README_PL.md) - Główny przewodnik TensorFlow
- 🤖 [UNITREE_G1_GUIDE_PL.md](../UNITREE_G1_GUIDE_PL.md) - Przewodnik dla robota G1
- 🔧 [ROBOTICS_APPLICATIONS_PL.md](../ROBOTICS_APPLICATIONS_PL.md) - Zastosowania w robotyce

### Kursy online
- [TensorFlow w praktyce](https://www.coursera.org/specializations/tensorflow-in-practice)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai](https://www.fast.ai/) - Praktyczne podejście do DL

### Społeczności
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [r/robotics](https://reddit.com/r/robotics)

## Podsumowanie

Przykłady w tym katalogu to:
- ✅ Świetny punkt startowy do nauki TensorFlow
- ✅ Demonstracja praktycznych zastosowań
- ✅ Baza dla projektów z robotem Unitree G1
- ⚠️ Mogą być przestarzałe (preferuj Python API)

**Najważniejsze przykłady dla robotyki:**
1. 🥇 **label_image** - Rozpoznawanie obiektów
2. 🥈 **speech_commands** - Sterowanie głosowe
3. 🥉 **image_retraining** - Dostosowanie modeli

**Następne kroki:**
1. Przejdź przez `label_image/README_PL.md`
2. Uruchom przykłady na własnych danych
3. Zobacz `UNITREE_G1_GUIDE_PL.md` dla integracji z robotem

---

**Powodzenia w nauce i projektach! 🚀🤖**

*Pytania? Sprawdź [ROBOTICS_APPLICATIONS_PL.md](../ROBOTICS_APPLICATIONS_PL.md) lub [GitHub Issues](https://github.com/tensorflow/tensorflow/issues)*
