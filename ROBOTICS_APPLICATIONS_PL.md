# TensorFlow w Robotyce - Podsumowanie i Zastosowania
## Dokument dla projektu z robotem Unitree G1 EDU-U6

### Politechnika Rzeszowska - Laboratorium AI i Robotyki

---

## 1. Wprowadzenie

### 1.1 Czym jest TensorFlow?

TensorFlow to otwarta platforma do uczenia maszynowego (machine learning) stworzona przez Google. W kontekście robotyki TensorFlow pozwala robotom:

- **"Widzieć"** - rozpoznawać obiekty, twarze, gesty
- **"Słyszeć"** - rozumieć polecenia głosowe
- **"Myśleć"** - podejmować inteligentne decyzje
- **"Uczyć się"** - doskonalić swoje umiejętności na podstawie doświadczeń

### 1.2 Dlaczego TensorFlow dla robota Unitree G1 EDU-U6?

Robot humanoidalny Unitree G1 EDU-U6 to zaawansowana platforma edukacyjna, która idealnie nadaje się do integracji z TensorFlow:

| Możliwość robota | Zastosowanie TensorFlow | Korzyść |
|------------------|------------------------|---------|
| Kamery HD | Widzenie komputerowe | Rozpoznawanie obiektów, nawigacja wizualna |
| Mikrofony | Rozpoznawanie mowy | Sterowanie głosowe, interakcja z użytkownikiem |
| Manipulatory | Planowanie chwytów | Inteligentna manipulacja obiektami |
| Sensory | Predykcja ruchu | Bezpieczna nawigacja, unikanie przeszkód |
| Procesor | Edge computing | Przetwarzanie w czasie rzeczywistym |

---

## 2. Główne zastosowania TensorFlow w projekcie

### 2.1 Widzenie Komputerowe (Computer Vision)

#### A. Rozpoznawanie i klasyfikacja obiektów

**Technologia:** Konwolucyjne sieci neuronowe (CNN)

**Modele:**
- **Inception V3** - Wysoka dokładność, 1000 kategorii obiektów
- **MobileNet** - Szybki, optymalny dla robotów
- **EfficientDet** - Detekcja wielu obiektów jednocześnie

**Przykładowy kod (z tego repozytorium):**
```python
# tensorflow/examples/label_image/label_image.py
graph = load_graph("inception_v3.pb")
image = read_tensor_from_image_file("robot_view.jpg")
results = classify_image(graph, image)
# Robot wie co widzi: "człowiek", "piłka", "krzesło"...
```

**Zastosowania dla G1:**
- Robot rozpoznaje co ma podnieść
- Identyfikuje ludzi w pomieszczeniu
- Wykrywa przeszkody na swojej drodze
- Czyta napisy i symbole

**Przykładowy scenariusz:**
```
1. Robot skanuje pomieszczenie kamerą
2. TensorFlow rozpoznaje: "kubek", "książka", "telefon"
3. Użytkownik mówi: "podaj mi kubek"
4. Robot lokalizuje kubek, planuje chwyt i podnosi go
```

#### B. Segmentacja obrazu

**Cel:** Rozdzielenie obrazu na regiony (co gdzie jest)

**Zastosowania:**
- Oddzielenie obiektów od tła
- Identyfikacja powierzchni do chodzenia
- Wykrywanie granic przeszkód

#### C. Śledzenie obiektów

**Cel:** Monitorowanie pozycji obiektu w czasie

**Zastosowania:**
- Robot podąża za osobą
- Śledzi piłkę aby ją złapać
- Monitoruje ruch w pomieszczeniu

### 2.2 Rozpoznawanie i Przetwarzanie Mowy

#### A. Rozpoznawanie poleceń głosowych (Keyword Spotting)

**Technologia:** Rekurencyjne sieci neuronowe (RNN/LSTM) lub CNN na spektrogramach

**Modele:**
- Speech Commands - Rozpoznawanie krótkich komend
- DeepSpeech - Pełna transkrypcja mowy
- Whisper - Najnowocześniejszy model od OpenAI

**Przykładowy kod (z tego repozytorium):**
```python
# tensorflow/examples/speech_commands/train.py
# Trening modelu na polskich komendach:
python train.py --wanted_words=idz,stop,lewo,prawo,podnies,poloz

# Rozpoznawanie w czasie rzeczywistym
model = load_model("speech_model.pb")
while robot.is_active():
    audio = microphone.record()
    command = recognize(model, audio)
    robot.execute(command)  # "idź", "stop", etc.
```

**Zastosowania dla G1:**
- Sterowanie ruchem: "idź naprzód", "zawróć", "stop"
- Kontrola manipulatora: "podnieś", "połóż", "otwórz chwyt"
- Tryby pracy: "tryb autonomiczny", "tryb manualny"
- Interakcja: "tak", "nie", "powtórz"

**Przykładowy scenariusz:**
```
Użytkownik: "G1, podejdź do stołu"
Robot: Rozpoznaje komendę "podejdź" + obiekt "stół"
       → Używa widzenia aby znaleźć stół
       → Planuje ścieżkę
       → Porusza się do celu
```

#### B. Synteza mowy (Text-to-Speech)

**Cel:** Robot mówi do użytkownika

**Zastosowania:**
- Potwierdzanie poleceń: "Rozumiem, idę do stołu"
- Informowanie o problemach: "Nie mogę znaleźć obiektu"
- Raportowanie stanu: "Bateria niska"

### 2.3 Nawigacja i Planowanie Ścieżki

#### A. SLAM (Simultaneous Localization and Mapping)

**Cel:** Robot buduje mapę otoczenia i wie gdzie się znajduje

**Komponenty TensorFlow:**
- CNN do ekstrahowania cech z obrazów
- Odometry prediction - przewidywanie ruchu
- Loop closure detection - rozpoznawanie znanego miejsca

**Zastosowania:**
- Autonomiczna eksploracja pomieszczenia
- Zapamiętywanie rozkładu przestrzeni
- Znajdowanie drogi powrotnej

#### B. Unikanie przeszkód

**Technologia:** Deep Q-Learning, Policy Gradients

**Jak działa:**
1. Robot widzi otoczenie (kamery + sensory)
2. Model TensorFlow ocenia bezpieczeństwo różnych ruchów
3. Robot wybiera najbezpieczniejszą akcję
4. Uczy się z doświadczenia (Reinforcement Learning)

### 2.4 Manipulacja Obiektami

#### A. Planowanie chwytów (Grasp Planning)

**Cel:** Określenie jak chwycić obiekt

**Proces:**
1. Rozpoznanie obiektu (co to jest?)
2. Segmentacja (gdzie dokładnie jest?)
3. Predykcja punktów chwytnych
4. Planowanie trajektorii manipulatora
5. Kontrola siły chwytu

**Model:** GraspNet, ContactGraspNet

#### B. Kontrola siły

**Cel:** Robot nie zniszczy delikatnych obiektów

**Zastosowania:**
- Podnoszenie jajek bez tłuczenia
- Podawanie kubka z wodą bez rozlania
- Uścisk dłoni odpowiedniej siły

---

## 3. Architektura Systemu dla Unitree G1

### 3.1 Schemat integracji

```
┌─────────────────────────────────────────────────────────────────┐
│                     ROBOT UNITREE G1 EDU-U6                      │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   PERCEPCJA  │  │  PRZETWARZANIE│  │      AKCJA           │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────┤  │
│  │              │  │              │  │                      │  │
│  │ • Kamery     │  │ TensorFlow   │  │ • Manipulatory       │  │
│  │ • Mikrofony  │─>│ Models:      │─>│ • Nogi (chód)        │  │
│  │ • IMU        │  │              │  │ • Głowa (obrót)      │  │
│  │ • LIDAR      │  │ • Vision     │  │ • Synteza mowy       │  │
│  │ • Dotyk      │  │ • Speech     │  │ • Sygnalizacja LED   │  │
│  │              │  │ • Navigation │  │                      │  │
│  │              │  │ • Control    │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            WARSTWA UCZENIA (Training Pipeline)             │ │
│  │  • Zbieranie danych z czujników                            │ │
│  │  • Etykietowanie (labeling)                                │ │
│  │  • Trening modeli offline                                  │ │
│  │  • Walidacja i testowanie                                  │ │
│  │  • Deployment na robota                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUKTURA ZEWNĘTRZNA                     │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  GPU Server     │  │  Cloud Storage  │  │  Monitoring     │ │
│  │  (Trening)      │  │  (Modele, dane) │  │  (Dashboard)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Przepływ danych

```
┌─────────────────────────────────────────────────────────────┐
│ PRZYKŁADOWY SCENARIUSZ: "Podaj mi butelkę"                  │
└─────────────────────────────────────────────────────────────┘

KROK 1: ROZPOZNAWANIE POLECENIA
┌──────────────────────────────────────┐
│ Mikrofon → TensorFlow Speech Model   │
│ Wynik: "podaj" + "butelka"           │
└──────────────────┬───────────────────┘
                   │
                   ▼
KROK 2: LOKALIZACJA OBIEKTU
┌──────────────────────────────────────┐
│ Kamera → TensorFlow Vision Model     │
│ Wynik: Butelka na stole, (x,y,z)     │
└──────────────────┬───────────────────┘
                   │
                   ▼
KROK 3: PLANOWANIE RUCHU
┌──────────────────────────────────────┐
│ TensorFlow + Klasyczne algorytmy     │
│ - Ścieżka do stołu (A*)              │
│ - Trajektoria manipulatora (MoveIt)  │
│ - Punkty chwytne (GraspNet)          │
└──────────────────┬───────────────────┘
                   │
                   ▼
KROK 4: WYKONANIE
┌──────────────────────────────────────┐
│ Kontrola niskopoziomowa               │
│ - Chód do stołu                       │
│ - Wyciągnięcie ręki                   │
│ - Chwyt butelki                       │
│ - Podanie użytkownikowi               │
└──────────────────┬───────────────────┘
                   │
                   ▼
KROK 5: POTWIERDZENIE
┌──────────────────────────────────────┐
│ Robot mówi: "Proszę"                  │
│ (TensorFlow TTS)                      │
└───────────────────────────────────────┘
```

---

## 4. Praktyczne przykłady projektów

### Projekt 1: Autonomiczny asystent domowy

**Cel:** Robot autonomicznie pomaga w domu

**Funkcje:**
1. Patrzy na stół i rozpoznaje przedmioty
2. Reaguje na polecenia: "przynieś telefon", "posprząta"j
3. Nawiguje po pomieszczeniu unikając przeszkód
4. Manipuluje obiektami

**Wymagane modele TensorFlow:**
- Object detection (YOLO/EfficientDet)
- Speech recognition (Speech Commands)
- SLAM/navigation
- Grasp planning

### Projekt 2: Robot społeczny (Social Robot)

**Cel:** Robot wchodzi w interakcje z ludźmi

**Funkcje:**
1. Rozpoznaje twarze (Face Recognition)
2. Wykrywa emocje (Emotion Detection)
3. Rozmawia z użytkownikiem (Speech + TTS)
4. Reaguje gestami i mimiką

**Wymagane modele TensorFlow:**
- Face detection & recognition
- Emotion classification
- Speech recognition + synthesis
- Gesture recognition

### Projekt 3: Robot edukacyjny

**Cel:** Uczy dzieci poprzez interakcję

**Funkcje:**
1. Pokazuje karty z obrazkami
2. Pyta "Co to jest?"
3. Rozpoznaje odpowiedź mowy
4. Potwierdza lub koryguje

**Wymagane modele TensorFlow:**
- Image classification
- Speech recognition
- Text-to-speech

---

## 5. Workflow: Od pomysłu do działającego robota

### Faza 1: Projektowanie (1-2 tygodnie)

```
1. Zdefiniuj funkcjonalność
   ↓
2. Wybierz modele TensorFlow
   ↓
3. Zaprojektuj architekturę systemu
   ↓
4. Określ wymagania sprzętowe
```

### Faza 2: Przygotowanie danych (2-4 tygodnie)

```
Dla Vision:
- Zbierz zdjęcia z kamery robota
- Etykietuj obiekty (labelimg, CVAT)
- Augmentacja danych

Dla Speech:
- Nagraj komendy (różni mówcy)
- Transkrypcja
- Augmentacja (szum, echo)
```

### Faza 3: Trening modeli (1-4 tygodnie)

```
1. Transfer learning od pretrained modeli
   ↓
2. Fine-tuning na własnych danych
   ↓
3. Walidacja i optymalizacja
   ↓
4. Konwersja do TensorFlow Lite
```

### Faza 4: Integracja (2-3 tygodnie)

```
1. Implementacja API do robota
   ↓
2. Testowanie na symulatorze
   ↓
3. Deployment na prawdziwym robocie
   ↓
4. Testowanie end-to-end
```

### Faza 5: Optymalizacja (ciągła)

```
- Zbieranie nowych danych w realnych warunkach
- Retraining modeli
- Deployment aktualizacji
- Monitoring performance
```

---

## 6. Narzędzia i zasoby

### 6.1 Do treningu modeli

| Narzędzie | Zastosowanie | Link |
|-----------|--------------|------|
| **TensorFlow** | Framework główny | tensorflow.org |
| **Keras** | API wysokopoziomowe | keras.io |
| **TensorBoard** | Wizualizacja treningu | tensorflow.org/tensorboard |
| **Google Colab** | Darmowe GPU do treningu | colab.research.google.com |

### 6.2 Do zbierania i etykietowania danych

| Narzędzie | Zastosowanie |
|-----------|--------------|
| **LabelImg** | Etykietowanie obrazów |
| **CVAT** | Annotacja wideo i obrazów |
| **Audacity** | Edycja nagrań audio |
| **RoboFlow** | Zarządzanie datasetami |

### 6.3 Gotowe modele (pretrained)

| Model | Zadanie | Źródło |
|-------|---------|--------|
| **MobileNet** | Klasyfikacja obrazów | TensorFlow Hub |
| **YOLOv5** | Detekcja obiektów | Ultralytics |
| **Speech Commands** | Komendy głosowe | TensorFlow |
| **DeepSpeech** | Rozpoznawanie mowy | Mozilla |

---

## 7. Najlepsze praktyki

### 7.1 Wydajność

✅ **Używaj TensorFlow Lite** na robocie (3-5x szybsze)
✅ **Kwantyzacja** modeli (int8) dla mniejszego rozmiaru
✅ **Batch inference** gdy możliwe
✅ **GPU/TPU** dla treningu, CPU/Edge TPU dla inferencji
❌ Unikaj pełnego TensorFlow na robocie (za ciężki)

### 7.2 Dokładność

✅ **Transfer learning** od dużych modeli pretrained
✅ **Data augmentation** zwiększa robustness
✅ **Ensemble** modeli dla krytycznych zadań
✅ **Continuous learning** z nowych danych
❌ Nie trenuj od zera jeśli nie musisz

### 7.3 Bezpieczeństwo

✅ **Fallback** gdy model niepewny (confidence < 0.7)
✅ **Emergency stop** zawsze dostępny
✅ **Sanity checks** na predykcje
✅ **Human-in-the-loop** dla krytycznych decyzji
❌ Nigdy nie ufaj modelowi w 100%

---

## 8. Troubleshooting - Częste problemy

### Problem: Model działa wolno na robocie

**Rozwiązania:**
1. Konwertuj do TensorFlow Lite
2. Użyj kwantyzacji (int8)
3. Wybierz mniejszy model (MobileNet zamiast ResNet)
4. Zmniejsz rozdzielczość wejściową
5. Użyj Edge TPU jeśli dostępny

### Problem: Niska accuracy

**Rozwiązania:**
1. Zbierz więcej danych treningowych
2. Popraw jakość etykiet
3. Użyj data augmentation
4. Wypróbuj większy/lepszy model
5. Fine-tuning dłużej

### Problem: Model się "przeuczył" (overfitting)

**Rozwiązania:**
1. Zwiększ zbiór treningowy
2. Użyj regularizacji (L2, dropout)
3. Data augmentation
4. Early stopping podczas treningu
5. Uproszczenie modelu

---

## 9. Podsumowanie

### Kluczowe wnioski

1. **TensorFlow to potężne narzędzie** dla robotyki edukacyjnej
2. **Unitree G1 EDU-U6** idealnie nadaje się do projektów z AI
3. **Gotowe modele** pozwalają szybko startować
4. **Własne dane** dają najlepsze rezultaty dla specific tasks
5. **Praktyka** jest kluczowa - eksperymentuj!

### Co dalej?

#### Dla początkujących:
1. ✅ Przejdź przez przykłady w `tensorflow/examples/`
2. ✅ Uruchom label_image.py z własnymi zdjęciami
3. ✅ Wytrenuj model speech_commands na polskich słowach
4. ✅ Zintegruj z symulatorem robota

#### Dla średnio zaawansowanych:
1. ✅ Zbierz własny dataset z kamery robota
2. ✅ Wytrenuj model detekcji obiektów
3. ✅ Zaimplementuj prosty SLAM
4. ✅ Stwórz kompletny system sterowania głosem

#### Dla zaawansowanych:
1. ✅ Reinforcement Learning do kontroli robota
2. ✅ Multi-task learning (vision + speech jednocześnie)
3. ✅ Real-time SLAM z deep learning
4. ✅ Publish paper o swoich wynikach!

---

## 10. Dodatkowe zasoby

### Dokumentacja w tym repozytorium (w języku polskim):
- `README_PL.md` - Ogólny przewodnik TensorFlow
- `UNITREE_G1_GUIDE_PL.md` - Szczegółowy przewodnik dla robota G1
- `tensorflow/examples/label_image/README_PL.md` - Rozpoznawanie obrazów
- `tensorflow/examples/speech_commands/README_PL.md` - Rozpoznawanie mowy

### Kursy online:
- TensorFlow w praktyce (Coursera)
- Deep Learning Specialization (Coursera)
- Fast.ai - Practical Deep Learning for Coders

### Społeczności:
- TensorFlow Forum: discuss.tensorflow.org
- r/MachineLearning (Reddit)
- r/robotics (Reddit)
- Lokalna grupa AI/ML na Politechnice Rzeszowskiej

### Książki (polecane):
- "Hands-On Machine Learning" - Aurélien Géron
- "Deep Learning" - Ian Goodfellow
- "Programming Robots with ROS" - Morgan Quigley

---

**Życzymy powodzenia w Waszych projektach z robotem Unitree G1 EDU-U6!**

**Zespół Laboratorium AI i Robotyki**
**Politechnika Rzeszowska**

🤖 + 🧠 = 🚀
