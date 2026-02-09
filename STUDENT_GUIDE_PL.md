# 📚 Przewodnik Studenta - TensorFlow dla Robotyki
## Politechnika Rzeszowska - Projekt Robot Unitree G1 EDU-U6

---

Witaj! Ten dokument jest Twoim przewodnikiem po zasobach TensorFlow przygotowanych specjalnie dla projektu z robotem humanoidalnym **Unitree G1 EDU-U6**.

## 🎯 Szybki start - Od czego zacząć?

### Jesteś tu pierwszy raz?
1. 📖 Przeczytaj [README_PL.md](README_PL.md) - Ogólne wprowadzenie do TensorFlow
2. 🤖 Zobacz [UNITREE_G1_GUIDE_PL.md](UNITREE_G1_GUIDE_PL.md) - Specyfika robota G1
3. 🚀 Przejdź do [pierwszego przykładu](#-pierwszy-przykład---rozpoznawanie-obrazów)

### Masz już podstawy?
1. 📊 Sprawdź [ROBOTICS_APPLICATIONS_PL.md](ROBOTICS_APPLICATIONS_PL.md) - Zaawansowane zastosowania
2. 💻 Wybierz projekt który Cię interesuje
3. 🔨 Zacznij kodować!

---

## 📑 Spis treści - Wszystkie zasoby

### 📘 Dokumentacja główna

| Dokument | Opis | Dla kogo | Czas czytania |
|----------|------|----------|---------------|
| [README_PL.md](README_PL.md) | Wprowadzenie do TensorFlow i podstawy | Wszyscy | 15 min |
| [UNITREE_G1_GUIDE_PL.md](UNITREE_G1_GUIDE_PL.md) | Szczegółowy przewodnik integracji z robotem | Średnio zaawansowani | 45 min |
| [ROBOTICS_APPLICATIONS_PL.md](ROBOTICS_APPLICATIONS_PL.md) | Kompleksowe zastosowania w robotyce | Zaawansowani | 60 min |

### 💡 Przykłady z kodem

| Przykład | Technologia | Dokumentacja | Kod |
|----------|-------------|--------------|-----|
| **Rozpoznawanie obrazów** | Computer Vision | [📖 README](tensorflow/examples/label_image/README_PL.md) | [🐍 Python](tensorflow/examples/label_image/label_image.py) |
| **Polecenia głosowe** | Speech Recognition | [📖 README](tensorflow/examples/speech_commands/README_PL.md) | [🐍 Python](tensorflow/examples/speech_commands/train.py) |
| **Przegląd przykładów** | Ogólnie | [📖 README](tensorflow/examples/README_PL.md) | [📁 Katalog](tensorflow/examples/) |

---

## 🎓 Ścieżki nauki

### 🟢 Poziom 1: Początkujący (2-4 tygodnie)

**Cel:** Zrozumienie podstaw TensorFlow i uruchomienie pierwszych przykładów.

```
Tydzień 1: Teoria
├─ Przeczytaj README_PL.md
├─ Zainstaluj TensorFlow
└─ Uruchom pierwszy przykład (label_image.py)

Tydzień 2: Praktyka - Vision
├─ Zrozum kod label_image.py (czytaj komentarze!)
├─ Przetestuj na własnych zdjęciach
└─ Eksperymentuj z parametrami

Tydzień 3-4: Praktyka - Speech
├─ Przejdź przez speech_commands
├─ Nagraj własne polskie komendy (10-20 nagrań)
├─ Wytrenuj prosty model
└─ Przetestuj rozpoznawanie
```

**Sprawdź swoją wiedzę:**
- ✅ Potrafisz uruchomić przykład label_image.py
- ✅ Rozumiesz co to jest tensor i jak wygląda
- ✅ Wiesz jak przetwarzać obraz przed podaniem do sieci
- ✅ Potrafisz nagrać i przetworzyć audio

### 🟡 Poziom 2: Średnio zaawansowany (4-8 tygodni)

**Cel:** Trening własnych modeli i integracja z robotem (w symulacji).

```
Tydzień 1-2: Zbieranie danych
├─ Zbierz zbiór zdjęć z kamery robota (lub symulacji)
├─ Etykietuj dane (LabelImg, CVAT)
└─ Przygotuj polskie nagrania głosowe (100+ na słowo)

Tydzień 3-4: Trening modeli
├─ Fine-tuning modelu vision na swoich danych
├─ Trening modelu speech na polskich komendach
└─ Walidacja i optymalizacja

Tydzień 5-6: Deployment
├─ Konwersja do TensorFlow Lite
├─ Integracja z API robota (symulator)
└─ Testowanie end-to-end

Tydzień 7-8: Projekt
├─ Wybierz projekt (asystent, social robot, etc.)
├─ Implementuj pełny pipeline
└─ Dokumentuj wyniki
```

**Sprawdź swoją wiedzę:**
- ✅ Potrafisz wytrenować model na własnych danych
- ✅ Rozumiesz metryki (accuracy, loss, confusion matrix)
- ✅ Potrafisz zoptymalizować model dla robota
- ✅ Umiesz debugować problemy z modelem

### 🔴 Poziom 3: Zaawansowany (8+ tygodni)

**Cel:** Własne architektury, research, publikacje.

```
├─ Implementacja własnych architektur sieci
├─ Multi-task learning (vision + speech + control)
├─ Reinforcement Learning dla kontroli robota
├─ Real-time SLAM z deep learning
├─ Eksperymentowanie z nowymi technikami
└─ Publikacja wyników (paper, blog, prezentacja)
```

**Sprawdź swoją wiedzę:**
- ✅ Potrafisz zaprojektować własną architekturę sieci
- ✅ Rozumiesz backpropagation i optymalizację
- ✅ Implementujesz custom operations w TensorFlow
- ✅ Przyczyniasz się do open source / publikujesz research

---

## 🎯 Pierwszy przykład - Rozpoznawanie obrazów

### Krok 1: Instalacja (5 minut)

```bash
# Zainstaluj TensorFlow
pip install tensorflow numpy pillow

# Sprawdź instalację
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### Krok 2: Pobierz model (2 minuty)

```bash
cd tensorflow/examples/label_image

# Pobierz model Inception V3 (~90MB)
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -xz -C data/
```

### Krok 3: Uruchom przykład (1 minuta)

```bash
# Test na domyślnym obrazie
python3 label_image.py

# Test na własnym obrazie
python3 label_image.py --image=twoje_zdjecie.jpg
```

### Krok 4: Zrozum kod (30 minut)

Otwórz [label_image.py](tensorflow/examples/label_image/label_image.py) i przeczytaj komentarze. Każda funkcja ma szczegółowe wyjaśnienie PO POLSKU!

**Kluczowe funkcje:**
- `load_graph()` - Ładowanie modelu
- `read_tensor_from_image_file()` - Przetwarzanie obrazu
- `load_labels()` - Wczytywanie nazw kategorii
- Główna pętla - Klasyfikacja i wyświetlenie wyników

---

## 🎤 Drugi przykład - Rozpoznawanie mowy

### Przygotowanie (10 minut)

```bash
cd tensorflow/examples/speech_commands

# Instalacja dodatkowych bibliotek
pip install pyaudio scipy
```

### Trening na domyślnych danych (2-4 godziny)

```bash
# Dataset pobierze się automatycznie (~1GB)
python3 train.py
```

### Trening na polskich komendach (przygotowanie: 2-3 godziny)

#### 1. Nagraj dane

Dla każdej komendy nagraj 100+ przykładów (WAV, 16kHz, 1s, mono):

```
moje_komendy/
  idz/        <- "idź" (100 plików)
  stop/       <- "stop" (100 plików)  
  lewo/       <- "lewo" (100 plików)
  prawo/      <- "prawo" (100 plików)
  inne/       <- inne dźwięki (100 plików)
```

#### 2. Trenuj model

```bash
python3 train.py \
  --data_dir=moje_komendy \
  --wanted_words=idz,stop,lewo,prawo
```

#### 3. Testuj

```bash
python3 label_wav.py --wav=test.wav
```

---

## 🤖 Integracja z robotem Unitree G1

### Architektura systemu

```
┌──────────────────────────────────────────────────┐
│              ROBOT UNITREE G1                     │
│                                                    │
│  ┌────────────┐        ┌────────────┐            │
│  │  Kamery    │───────>│ TensorFlow │            │
│  │  Mikrofony │───────>│   Models   │            │
│  └────────────┘        └─────┬──────┘            │
│                              │                    │
│                              ▼                    │
│                        ┌─────────────┐            │
│                        │   Decyzje   │            │
│                        │   AI/ML     │            │
│                        └──────┬──────┘            │
│                               │                   │
│                               ▼                   │
│                        ┌─────────────┐            │
│                        │  Sterowanie │            │
│                        │  • Ruch     │            │
│                        │  • Chwyt    │            │
│                        │  • Mowa     │            │
│                        └─────────────┘            │
└──────────────────────────────────────────────────┘
```

### Przykładowy kod integracji

Zobacz szczegóły w [UNITREE_G1_GUIDE_PL.md](UNITREE_G1_GUIDE_PL.md)

---

## 📚 Dodatkowe materiały

### Kursy online (angielski)
- [TensorFlow in Practice](https://www.coursera.org/specializations/tensorflow-in-practice) - Coursera
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Andrew Ng
- [Fast.ai](https://www.fast.ai/) - Praktyczne podejście

### Książki (polecane)
- "Hands-On Machine Learning" - Aurélien Géron
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio
- "Deep Learning with Python" - François Chollet

### Społeczności
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [r/robotics](https://reddit.com/r/robotics)
- Stack Overflow - tag `tensorflow`

### Narzędzia
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Wizualizacja
- [TensorFlow Hub](https://tfhub.dev/) - Gotowe modele
- [Google Colab](https://colab.research.google.com/) - Darmowe GPU

---

## ❓ FAQ - Często zadawane pytania

### Q: Czy muszę znać C++?
**A:** Nie! Wszystkie przykłady działają w Pythonie. C++ jest opcjonalne.

### Q: Ile czasu zajmie trening modelu?
**A:** 
- Na CPU: 2-4 godziny (speech commands)
- Na GPU: 30-60 minut
- Transfer learning: 10-30 minut

### Q: Czy mogę trenować na polskich danych?
**A:** Tak! Przykład speech_commands działa z dowolnymi słowami, również polskimi.

### Q: Jak zoptymalizować model dla robota?
**A:** Użyj TensorFlow Lite + kwantyzacja. Zobacz [UNITREE_G1_GUIDE_PL.md](UNITREE_G1_GUIDE_PL.md#optymalizacja-dla-robota)

### Q: Co jeśli model ma niską dokładność?
**A:** 
1. Zbierz więcej danych (najważniejsze!)
2. Użyj data augmentation
3. Fine-tune dłużej
4. Wypróbuj lepszy model

### Q: Gdzie znajdę więcej przykładów?
**A:** 
- [TensorFlow Examples](https://github.com/tensorflow/examples)
- [TensorFlow Models](https://github.com/tensorflow/models)
- [Papers with Code](https://paperswithcode.com/)

---

## 🎯 Projekty do realizacji

### Projekt 1: Asystent rozpoznający obiekty ⭐
**Trudność:** Łatwy  
**Czas:** 2-3 tygodnie  
**Opis:** Robot rozpoznaje przedmioty na stole i informuje użytkownika.

### Projekt 2: Sterowanie głosowe ⭐⭐
**Trudność:** Średni  
**Czas:** 3-4 tygodnie  
**Opis:** Robot reaguje na polskie polecenia głosowe.

### Projekt 3: Autonomiczne dostarczanie obiektów ⭐⭐⭐
**Trudność:** Zaawansowany  
**Czas:** 6-8 tygodni  
**Opis:** Robot znajduje, podnosi i dostarcza wskazany obiekt.

### Projekt 4: Robot społeczny ⭐⭐⭐
**Trudność:** Zaawansowany  
**Czas:** 8-10 tygodni  
**Opis:** Robot rozpoznaje twarze, emocje i prowadzi konwersację.

---

## 📞 Pomoc i wsparcie

### Problemy techniczne?
1. Sprawdź sekcję "Troubleshooting" w odpowiednim README
2. Przeszukaj [GitHub Issues](https://github.com/tensorflow/tensorflow/issues)
3. Zapytaj na [TensorFlow Forum](https://discuss.tensorflow.org/)

### Pytania o projekt?
- Skontaktuj się z prowadzącym laboratorium
- Współpracuj z innymi studentami
- Dokumentuj swoje rozwiązania

### Znalazłeś błąd w dokumentacji?
- Otwórz Issue na GitHubie
- Zaproponuj poprawkę (Pull Request)

---

## ✅ Checklist studenta

Przed rozpoczęciem pracy:
- [ ] Przeczytałem README_PL.md
- [ ] Zainstalowałem TensorFlow i zależności
- [ ] Uruchomiłem przykład label_image.py
- [ ] Zrozumiałem podstawy sieci neuronowych

Przed integracją z robotem:
- [ ] Wytrenowałem własny model
- [ ] Przetestowałem na różnych danych
- [ ] Zoptymalizowałem dla czasu rzeczywistego
- [ ] Przeczytałem UNITREE_G1_GUIDE_PL.md

Przed zakończeniem projektu:
- [ ] Kod jest skomentowany
- [ ] Dokumentacja jest kompletna
- [ ] Testy przeszły pomyślnie
- [ ] Projekt działa na prawdziwym robocie

---

## 🎓 Podsumowanie

### Pamiętaj:
1. **Praktyka czyni mistrza** - Eksperymentuj!
2. **Dokumentuj wszystko** - Przyszłe ty będzie wdzięczne
3. **Dziel się wiedzą** - Pomóż innym studentom
4. **Nie bój się błędów** - To najlepsza metoda nauki

### Sukces to:
- ✨ Zrozumienie jak działa TensorFlow
- ✨ Umiejętność trenowania własnych modeli
- ✨ Integracja AI z robotem
- ✨ Radość z działającego projektu!

---

**Powodzenia w Waszej przygodzie z TensorFlow i robotyką!** 🚀🤖

---

*Ostatnia aktualizacja: 2024*  
*Politechnika Rzeszowska - Laboratorium AI i Robotyki*  
*Projekt: Robot Humanoidalny Unitree G1 EDU-U6*
