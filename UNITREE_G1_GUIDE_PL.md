# Przewodnik TensorFlow dla Robota Unitree G1 EDU-U6

## Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Architektura systemu](#architektura-systemu)
3. [Konfiguracja środowiska](#konfiguracja-środowiska)
4. [Praktyczne zastosowania](#praktyczne-zastosowania)
5. [Przykładowe projekty](#przykładowe-projekty)
6. [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## Wprowadzenie

### O robocie Unitree G1 EDU-U6

Unitree G1 EDU-U6 to zaawansowany robot humanoidalny zaprojektowany do celów edukacyjnych i badawczych. Posiada:
- **Kamery** - do percepcji wizualnej (widzenie komputerowe)
- **Mikrofony** - do rozpoznawania mowy
- **Manipulatory** - do interakcji z obiektami
- **Sensory** - do nawigacji i równoważenia

### Dlaczego TensorFlow?

TensorFlow jest idealnym wyborem dla robota G1, ponieważ:
- **Wydajność** - Optymalizowany dla różnych platform (CPU, GPU, TPU)
- **TensorFlow Lite** - Wersja dla urządzeń embedded, idealna dla robotyki
- **Gotowe modele** - Przedtrenowane sieci neuronowe do natychmiastowego użycia
- **Wsparcie społeczności** - Szeroka baza wiedzy i przykładów

## Architektura systemu

### Schemat integracji TensorFlow z robotem G1

```
┌─────────────────────────────────────────────────────────┐
│                   Robot Unitree G1 EDU-U6                │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Kamery    │  │   Mikrofony  │  │    Sensory      │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘ │
│         │                │                   │          │
│         └────────────────┼───────────────────┘          │
│                          │                              │
│                          ▼                              │
│         ┌────────────────────────────────┐              │
│         │   TensorFlow / TensorFlow Lite │              │
│         │                                │              │
│         │  • Rozpoznawanie obrazów       │              │
│         │  • Przetwarzanie mowy          │              │
│         │  • Podejmowanie decyzji        │              │
│         └────────────┬───────────────────┘              │
│                      │                                  │
│                      ▼                                  │
│         ┌────────────────────────────┐                  │
│         │  Sterowanie robotem        │                  │
│         │  • Manipulatory            │                  │
│         │  • Nawigacja               │                  │
│         │  • Interakcja              │                  │
│         └────────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

### Przepływ danych

1. **Wejście** - Dane z kamer i mikrofonów
2. **Przetwarzanie** - Modele TensorFlow analizują dane
3. **Decyzja** - Wyniki analizy określają akcje
4. **Wyjście** - Robot wykonuje odpowiednie ruchy i działania

## Konfiguracja środowiska

### Wymagania sprzętowe

**Minimalne:**
- Procesor: 4-rdzeniowy CPU
- RAM: 8 GB
- Dysk: 20 GB wolnego miejsca

**Zalecane dla treningu modeli:**
- GPU: NVIDIA z obsługą CUDA (np. RTX 3060 lub lepszy)
- RAM: 16 GB lub więcej
- Dysk: SSD z 50 GB wolnego miejsca

### Instalacja na systemie robota

```bash
# 1. Aktualizacja systemu
sudo apt update && sudo apt upgrade -y

# 2. Instalacja Pythona i pip (jeśli nie ma)
sudo apt install python3 python3-pip -y

# 3. Instalacja TensorFlow Lite (lekka wersja dla robotyki)
pip3 install tflite-runtime

# 4. Lub pełna wersja TensorFlow (jeśli wystarczająca moc obliczeniowa)
pip3 install tensorflow

# 5. Dodatkowe biblioteki dla robotyki
pip3 install numpy opencv-python pillow
```

### Testowanie instalacji

```python
# test_tensorflow.py
import tensorflow as tf
print("Wersja TensorFlow:", tf.__version__)
print("GPU dostępne:", tf.config.list_physical_devices('GPU'))

# Test prostej operacji
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print("Test operacji:", tf.add(a, b).numpy())
```

```bash
python3 test_tensorflow.py
```

## Praktyczne zastosowania

### 1. Rozpoznawanie obiektów (Object Detection)

**Cel:** Robot rozpoznaje przedmioty w swoim otoczeniu.

**Przypadki użycia:**
- Identyfikacja obiektów do podniesienia
- Rozpoznawanie narzędzi
- Wykrywanie ludzi i przeszkód

**Implementacja:**

```python
# object_detection_for_g1.py
"""
Rozpoznawanie obiektów dla robota Unitree G1
Wykorzystuje przedtrenowany model Inception V3
"""

import tensorflow as tf
import numpy as np
from PIL import Image

# Krok 1: Ładowanie przedtrenowanego modelu
# Model Inception V3 został wytrenowany na 1000 kategorii obiektów
def load_model(model_path):
    """
    Ładuje zamrożony graf TensorFlow z modelem.
    
    Args:
        model_path: ścieżka do pliku .pb z modelem
    
    Returns:
        Załadowany graf TensorFlow
    """
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    with graph.as_default():
        tf.import_graph_def(graph_def)
    
    return graph

# Krok 2: Przetwarzanie obrazu z kamery robota
def preprocess_image(image_path, target_size=(299, 299)):
    """
    Przygotowuje obraz do analizy przez sieć neuronową.
    
    Dlaczego te kroki:
    - Zmiana rozmiaru: Model wymaga obrazów 299x299 pikseli
    - Normalizacja: Wartości pikseli muszą być w zakresie [0, 1]
    
    Args:
        image_path: ścieżka do obrazu z kamery
        target_size: rozmiar wymagany przez model
    
    Returns:
        Przetworzony obraz gotowy do analizy
    """
    # Wczytanie obrazu
    img = Image.open(image_path)
    
    # Zmiana rozmiaru do wymaganego przez model
    img = img.resize(target_size)
    
    # Konwersja do tablicy numpy
    img_array = np.array(img)
    
    # Dodanie wymiaru batch (model oczekuje wielu obrazów naraz)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalizacja wartości pikseli do zakresu [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

# Krok 3: Rozpoznawanie obiektów
def detect_objects(graph, image_array, labels_path):
    """
    Uruchamia model na obrazie i zwraca wyniki.
    
    Args:
        graph: załadowany model TensorFlow
        image_array: przetworzony obraz
        labels_path: ścieżka do pliku z etykietami
    
    Returns:
        Lista rozpoznanych obiektów z prawdopodobieństwem
    """
    # Wczytanie etykiet (nazw kategorii)
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Uruchomienie modelu
    with tf.compat.v1.Session(graph=graph) as sess:
        # Pobranie warstw wejściowej i wyjściowej
        input_tensor = graph.get_tensor_by_name('import/input:0')
        output_tensor = graph.get_tensor_by_name('import/InceptionV3/Predictions/Reshape_1:0')
        
        # Wykonanie predykcji
        predictions = sess.run(output_tensor, {input_tensor: image_array})
    
    # Przetworzenie wyników
    predictions = np.squeeze(predictions)  # Usunięcie niepotrzebnych wymiarów
    
    # Pobranie top 5 najbardziej prawdopodobnych obiektów
    top_5_indices = predictions.argsort()[-5:][::-1]
    
    results = []
    for i in top_5_indices:
        results.append({
            'obiekt': labels[i],
            'pewność': float(predictions[i])
        })
    
    return results

# Przykład użycia dla robota G1
if __name__ == "__main__":
    # Ścieżki do modelu i etykiet
    MODEL_PATH = "models/inception_v3_frozen.pb"
    LABELS_PATH = "models/imagenet_labels.txt"
    
    # Załaduj model (raz na początku)
    print("Ładowanie modelu...")
    model = load_model(MODEL_PATH)
    print("Model załadowany!")
    
    # Symulacja obrazu z kamery robota
    # W prawdziwej aplikacji pobierasz obraz bezpośrednio z kamery G1
    image_path = "camera_capture.jpg"
    
    # Przetwórz obraz
    processed_image = preprocess_image(image_path)
    
    # Rozpoznaj obiekty
    results = detect_objects(model, processed_image, LABELS_PATH)
    
    # Wyświetl wyniki
    print("\nRozpoznane obiekty:")
    for result in results:
        print(f"  {result['obiekt']}: {result['pewność']*100:.2f}%")
    
    # Robot może teraz zareagować na wykryty obiekt
    # np. jeśli wykryto "filiżankę", robot może ją podnieść
```

### 2. Rozpoznawanie poleceń głosowych (Speech Recognition)

**Cel:** Robot reaguje na polecenia głosowe użytkownika.

**Przypadki użycia:**
- Sterowanie głosowe ("idź", "stop", "podnieś")
- Interakcja człowiek-robot
- Nawigacja na polecenia

**Implementacja:**

```python
# voice_commands_for_g1.py
"""
System rozpoznawania poleceń głosowych dla robota Unitree G1
Oparty na przykładzie speech_commands z TensorFlow
"""

import tensorflow as tf
import numpy as np
import pyaudio
import wave

# Konfiguracja audio
SAMPLE_RATE = 16000  # Częstotliwość próbkowania (Hz)
DURATION = 1         # Czas nagrywania (sekundy)

# Polecenia, które robot rozumie
COMMANDS = [
    'idz',           # Go / Move forward
    'stop',          # Stop
    'lewo',          # Left
    'prawo',         # Right
    'podnies',       # Pick up
    'poloz',         # Put down
    'tak',           # Yes
    'nie'            # No
]

class VoiceCommandRecognizer:
    """
    Klasa do rozpoznawania poleceń głosowych dla robota.
    
    Używa modelu wytrenowanego na zestawie danych speech_commands.
    """
    
    def __init__(self, model_path):
        """
        Inicjalizacja rozpoznawania głosu.
        
        Args:
            model_path: ścieżka do wytrenowanego modelu
        """
        print("Ładowanie modelu rozpoznawania głosu...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model załadowany!")
        
        # Inicjalizacja PyAudio do nagrywania
        self.audio = pyaudio.PyAudio()
    
    def record_audio(self):
        """
        Nagrywa krótki fragment audio z mikrofonu robota.
        
        Returns:
            Tablica numpy z danymi audio
        """
        # Otwórz strumień audio
        stream = self.audio.open(
            format=pyaudio.paInt16,      # 16-bit audio
            channels=1,                   # Mono
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024
        )
        
        print("Nagrywam polecenie...")
        frames = []
        
        # Nagraj DURATION sekund audio
        for i in range(0, int(SAMPLE_RATE / 1024 * DURATION)):
            data = stream.read(1024)
            frames.append(data)
        
        print("Nagrywanie zakończone.")
        
        # Zamknij strumień
        stream.stop_stream()
        stream.close()
        
        # Konwertuj do tablicy numpy
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        return audio_data
    
    def preprocess_audio(self, audio_data):
        """
        Przetwarza dane audio do formatu wymaganego przez model.
        
        Kroki przetwarzania:
        1. Normalizacja - Wartości w zakresie [-1, 1]
        2. Spektrogram - Przekształcenie do reprezentacji częstotliwościowej
        3. MFCC - Mel-Frequency Cepstral Coefficients (standardowa reprezentacja dla mowy)
        
        Args:
            audio_data: Surowe dane audio
        
        Returns:
            Przetworzony tensor gotowy do modelu
        """
        # Normalizacja
        audio_normalized = audio_data.astype(np.float32) / 32768.0
        
        # Konwersja do tensora TensorFlow
        audio_tensor = tf.constant(audio_normalized)
        
        # Tutaj normalnie obliczasz spektrogram lub MFCC
        # Dla uproszczenia używamy surowego audio
        # W prawdziwej implementacji użyj tf.signal.stft i tf.signal.mfccs_from_log_mel_spectrogram
        
        # Dodaj wymiar batch
        audio_tensor = tf.expand_dims(audio_tensor, 0)
        
        return audio_tensor
    
    def recognize_command(self):
        """
        Nagrywa audio i rozpoznaje polecenie.
        
        Returns:
            Rozpoznane polecenie jako string
        """
        # Nagraj audio z mikrofonu
        audio_data = self.record_audio()
        
        # Przetwórz audio
        processed_audio = self.preprocess_audio(audio_data)
        
        # Uruchom model
        predictions = self.model.predict(processed_audio)
        
        # Znajdź najbardziej prawdopodobne polecenie
        command_index = np.argmax(predictions[0])
        confidence = predictions[0][command_index]
        
        command = COMMANDS[command_index]
        
        print(f"Rozpoznano: {command} (pewność: {confidence*100:.1f}%)")
        
        return command, confidence
    
    def cleanup(self):
        """Zamknięcie zasobów audio."""
        self.audio.terminate()

# Przykład użycia dla robota G1
def main():
    """
    Główna pętla rozpoznawania poleceń głosowych.
    Robot nasłuchuje poleceń i reaguje na nie.
    """
    # Inicjalizacja rozpoznawania głosu
    recognizer = VoiceCommandRecognizer('models/speech_commands_model.h5')
    
    print("\n" + "="*50)
    print("Robot Unitree G1 - System rozpoznawania głosu")
    print("="*50)
    print("\nDostępne polecenia:")
    for cmd in COMMANDS:
        print(f"  - {cmd}")
    print("\nNaciśnij Ctrl+C aby zakończyć.\n")
    
    try:
        while True:
            # Nasłuchuj polecenia
            command, confidence = recognizer.recognize_command()
            
            # Reaguj tylko jeśli pewność > 70%
            if confidence > 0.7:
                print(f"\n>>> Wykonuję polecenie: {command}")
                
                # Tutaj dodaj kod sterujący robotem
                if command == 'idz':
                    print("Robot porusza się do przodu...")
                    # robot.move_forward()
                
                elif command == 'stop':
                    print("Robot zatrzymuje się...")
                    # robot.stop()
                
                elif command == 'lewo':
                    print("Robot obraca się w lewo...")
                    # robot.turn_left()
                
                elif command == 'prawo':
                    print("Robot obraca się w prawo...")
                    # robot.turn_right()
                
                elif command == 'podnies':
                    print("Robot podnosi obiekt...")
                    # robot.pick_up()
                
                elif command == 'poloz':
                    print("Robot kładzie obiekt...")
                    # robot.put_down()
                
                print()
            else:
                print(f"Polecenie niezrozumiałe (pewność tylko {confidence*100:.1f}%)")
            
            # Krótka pauza przed następnym nagraniem
            import time
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nZamykanie systemu...")
        recognizer.cleanup()
        print("Do widzenia!")

if __name__ == "__main__":
    main()
```

### 3. Nawigacja i unikanie przeszkód

**Cel:** Robot nawiguje autonomicznie, unikając przeszkód.

**Przypadki użycia:**
- Poruszanie się po pomieszczeniu
- Mapowanie otoczenia
- Planowanie ścieżki

**Kluczowe komponenty:**
- **Detekcja przeszkód** - Wykorzystanie kamer i modelu segmentacji obrazu
- **Mapowanie** - SLAM (Simultaneous Localization and Mapping)
- **Planowanie ścieżki** - Algorytmy pathfinding z uczeniem maszynowym

## Przykładowe projekty

### Projekt 1: Asystent do rozpoznawania obiektów

**Zadanie:** Robot rozpoznaje przedmioty na stole i informuje użytkownika.

**Kroki:**
1. Konfiguracja kamery robota
2. Załadowanie modelu Inception V3
3. Ciągłe przetwarzanie obrazu z kamery
4. Wyświetlanie rozpoznanych obiektów

### Projekt 2: Sterowanie głosowe podstawowymi ruchami

**Zadanie:** Robot reaguje na polecenia głosowe (idź, stop, lewo, prawo).

**Kroki:**
1. Trening modelu na polskich komendach
2. Integracja z mikrofonami robota
3. Implementacja logiki sterowania
4. Testowanie i dostrajanie

### Projekt 3: Autonomiczne dostarczanie obiektów

**Zadanie:** Robot odbiera przedmiot z punktu A i dostarcza do punktu B.

**Kroki:**
1. Rozpoznawanie obiektu docelowego
2. Planowanie ścieżki do obiektu
3. Chwytanie obiektu
4. Nawigacja do miejsca docelowego
5. Położenie obiektu

## Rozwiązywanie problemów

### Problem: Model działa zbyt wolno

**Przyczyna:** Niewystarczająca moc obliczeniowa.

**Rozwiązania:**
1. Użyj TensorFlow Lite zamiast pełnego TensorFlow
2. Wykorzystaj GPU jeśli dostępne
3. Zmniejsz rozdzielczość wejściowego obrazu
4. Użyj mniejszego modelu (np. MobileNet zamiast Inception)

### Problem: Niskie accuracy rozpoznawania

**Przyczyna:** Model nie pasuje do konkretnego przypadku użycia.

**Rozwiązania:**
1. Dostrajanie (fine-tuning) modelu na własnych danych
2. Zbieranie większej ilości danych treningowych
3. Augmentacja danych (obroty, przesunięcia, etc.)
4. Wybór lepszego modelu dla konkretnego zadania

### Problem: Rozpoznawanie głosu nie działa

**Przyczyna:** Zakłócenia, nieprawidłowa konfiguracja mikrofonu.

**Rozwiązania:**
1. Sprawdź konfigurację mikrofonu (sample rate, formaty)
2. Dodaj filtrowanie szumów
3. Zwiększ czas nagrywania próbki
4. Trenuj model na danych z szumem tła

## Podsumowanie

TensorFlow to potężne narzędzie, które otwiera przed robotem Unitree G1 EDU-U6 nieograniczone możliwości:
- **Percepcja** - Widzenie i słyszenie
- **Inteligencja** - Rozumienie i uczenie się
- **Autonomia** - Samodzielne podejmowanie decyzji

**Następne kroki:**
1. Przejdź przez przykładowy kod w `tensorflow/examples/`
2. Eksperymentuj z własnymi danymi
3. Buduj własne modele dla specyficznych zadań
4. Dziel się swoimi projektami ze społecznością!

---

**Pytania? Problemy?**
- Sprawdź [README_PL.md](README_PL.md) dla ogólnych informacji
- Zobacz przykłady kodu w `tensorflow/examples/`
- Odwiedź [forum TensorFlow](https://discuss.tensorflow.org/)

**Powodzenia w projektach z robotem Unitree G1!** 🤖✨
