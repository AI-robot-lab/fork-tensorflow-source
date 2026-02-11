# Rozpoznawanie Poleceń Głosowych - Przewodnik dla Studentów

## Wprowadzenie

Ten przykład pokazuje jak zbudować system rozpoznawania mowy wykorzystujący TensorFlow. Jest to fundamentalna technologia dla robota Unitree G1 EDU-U6, pozwalająca mu reagować na polecenia głosowe użytkownika.

## Co robi ten system?

System rozpoznaje **krótkie słowa-komendy** z ograniczonego słownika. Standardowo rozpoznaje słowa:
- **"yes"** (tak) / **"no"** (nie)
- **"up"** (góra) / **"down"** (dół)
- **"left"** (lewo) / **"right"** (prawo)
- **"on"** (włącz) / **"off"** (wyłącz)
- **"stop"** (stop) / **"go"** (idź)

### To NIE jest pełny system rozpoznawania mowy

**Co potrafi:**
- ✅ Rozpoznaje pojedyncze, krótkie słowa
- ✅ Działa w czasie rzeczywistym
- ✅ Można trenować na własnych słowach (również polskich!)
- ✅ Odpowiedni dla robotyki i urządzeń embedded

**Czego nie potrafi:**
- ❌ Rozpoznawanie pełnych zdań
- ❌ Transkrypcja długich wypowiedzi
- ❌ Rozpoznawanie mowy ciągłej

**Dla zaawansowanego rozpoznawania mowy** polecamy systemy takie jak Kaldi, Whisper lub Google Cloud Speech-to-Text.

## Zastosowania w robocie Unitree G1 EDU-U6

### Przykładowe scenariusze

**Scenariusz 1: Sterowanie ruchem robota**
```
Użytkownik: "idź"     → Robot porusza się do przodu
Użytkownik: "stop"    → Robot zatrzymuje się
Użytkownik: "lewo"    → Robot obraca się w lewo
Użytkownik: "prawo"   → Robot obraca się w prawo
```

**Scenariusz 2: Kontrola manipulatora**
```
Użytkownik: "podnieś" → Robot podnosi przedmiot
Użytkownik: "połóż"   → Robot kładzie przedmiot
Użytkownik: "otwórz"  → Robot otwiera chwyt
Użytkownik: "zamknij" → Robot zamyka chwyt
```

**Scenariusz 3: Interakcja z użytkownikiem**
```
Użytkownik: "tak"     → Robot potwierdza akcję
Użytkownik: "nie"     → Robot anuluje akcję
Użytkownik: "pomoc"   → Robot wyświetla dostępne komendy
```

## Struktura projektu

### Główne pliki

```
speech_commands/
├── train.py                    # Trenowanie modelu
├── freeze.py                   # Tworzenie wersji produkcyjnej modelu
├── input_data.py              # Ładowanie i przetwarzanie danych audio
├── models.py                  # Definicje architektur sieci neuronowych
├── label_wav.py               # Rozpoznawanie pojedynczego pliku audio
├── label_wav_dir.py           # Rozpoznawanie całego katalogu
├── recognize_commands.py      # Rozpoznawanie w czasie rzeczywistym
└── README_PL.md               # Ten plik
```

### Przepływ pracy

```
┌─────────────────────┐
│  1. Zbierz dane     │  Nagraj pliki .wav z komendami
│     audio           │  lub pobierz gotowy dataset
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Trenuj model    │  python train.py --wanted_words=idź,stop,lewo,prawo
│     (train.py)      │  
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Testuj model    │  python label_wav.py --wav=test.wav
│     (label_wav.py)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Wdróż na        │  Użyj zamrożonego modelu w robocie
│     robocie         │  dla rozpoznawania w czasie rzeczywistym
└─────────────────────┘
```

## Jak rozpocząć - Krok po kroku

### Krok 1: Przygotowanie środowiska

```bash
# Upewnij się że masz zainstalowany TensorFlow
pip install tensorflow numpy

# Przejdź do katalogu speech_commands
cd tensorflow/examples/speech_commands
```

### Krok 2: Trening z gotowym zestawem danych

Użyj domyślnego datasetu Speech Commands (automatycznie się pobierze):

```bash
# Podstawowy trening (używa domyślnych słów)
python train.py

# Trening z wybranymi słowami
python train.py --wanted_words=yes,no,up,down,left,right
```

**Co się dzieje podczas treningu:**
1. **Pobieranie danych** (pierwszym razem)
   - Dataset Speech Commands (~1GB)
   - Tysiące nagrań .wav po 1 sekundzie każde
   - Różni mówcy, różne warunki nagrania

2. **Preprocessing audio**
   - Konwersja audio do spektrogramów
   - Augmentacja danych (szum, przesunięcia czasowe)
   - Normalizacja

3. **Trening sieci neuronowej**
   - Kilka tysięcy iteracji
   - Na CPU: 2-4 godziny
   - Na GPU: 30-60 minut

4. **Walidacja**
   - Sprawdzanie accuracy na zbiorze walidacyjnym
   - Powinno osiągnąć >90% accuracy

**Przykładowy output:**
```
INFO:tensorflow:Step #100: rate 0.001000, accuracy = 12.0%, cross entropy = 2.589
INFO:tensorflow:Step #200: rate 0.001000, accuracy = 25.0%, cross entropy = 2.234
INFO:tensorflow:Step #500: rate 0.001000, accuracy = 47.0%, cross entropy = 1.876
...
INFO:tensorflow:Step #4000: rate 0.001000, accuracy = 91.0%, cross entropy = 0.334
```

### Krok 3: Trening na własnych polskich komendach

#### 3a. Przygotowanie danych

Utwórz strukturę katalogów:

```
moje_polskie_komendy/
├── idz/              # Polecenie "idź"
│   ├── nagranie001.wav
│   ├── nagranie002.wav
│   └── ... (minimum 100 nagrań)
├── stop/             # Polecenie "stop"
│   ├── nagranie101.wav
│   ├── nagranie102.wav
│   └── ...
├── lewo/             # Polecenie "lewo"
│   └── ...
├── prawo/            # Polecenie "prawo"
│   └── ...
└── inne/             # Inne dźwięki (szum, inne słowa)
    └── ...
```

**Wymagania dla nagrań:**
- **Format:** WAV, 16-bit PCM
- **Częstotliwość:** 16000 Hz (16 kHz)
- **Długość:** 1 sekunda
- **Kanały:** Mono (1 kanał)
- **Jakość:** Wyraźna wymowa, mało szumu tła

**Narzędzie do nagrywania:**

```bash
# Linux/Mac - nagrywanie z mikrofonu
arecord -f cd -d 1 -r 16000 nagranie.wav

# Lub użyj Audacity (darmowy edytor audio)
# 1. Nagraj słowo
# 2. Eksportuj jako WAV 16kHz mono
```

**Wskazówki dla lepszych wyników:**
- Nagraj minimum 100 przykładów każdego słowa
- Użyj różnych mówców (mężczyźni, kobiety, różne akcenty)
- Nagraj w różnych warunkach (ciche pomieszczenie, z lekkim szumem)
- Słowo powinno być wypowiadane w środku 1-sekundowego nagrania

#### 3b. Uruchomienie treningu

```bash
python train.py \
  --data_dir=moje_polskie_komendy \
  --wanted_words=idz,stop,lewo,prawo \
  --train_dir=/tmp/polski_speech_model \
  --how_many_training_steps=4000,1000 \
  --learning_rate=0.001,0.0001
```

**Parametry wyjaśnione:**
- `--data_dir`: Katalog z Twoimi nagraniami
- `--wanted_words`: Słowa które chcesz rozpoznawać (oddzielone przecinkami)
- `--train_dir`: Gdzie zapisać wytrenowany model
- `--how_many_training_steps`: Liczba kroków treningu (więcej = lepszy model, ale dłużej)
- `--learning_rate`: Szybkość uczenia (zaczyna od 0.001, potem spada do 0.0001)

### Krok 4: Testowanie wytrenowanego modelu

#### Test pojedynczego pliku

```bash
python label_wav.py \
  --wav=test_nagranie.wav \
  --graph=/tmp/polski_speech_model/frozen_graph.pb \
  --labels=/tmp/polski_speech_model/labels.txt
```

**Przykładowy wynik:**
```
Ładowanie modelu...
Przetwarzanie: test_nagranie.wav

Wyniki rozpoznawania:
  idz:    0.89  (89%)
  stop:   0.05  (5%)
  lewo:   0.03  (3%)
  prawo:  0.02  (2%)

Rozpoznane polecenie: idz (pewność: 89%)
```

#### Test całego katalogu

```bash
python label_wav_dir.py \
  --wav_dir=testy/ \
  --graph=/tmp/polski_speech_model/frozen_graph.pb \
  --labels=/tmp/polski_speech_model/labels.txt
```

### Krok 5: Zamrożenie modelu dla produkcji

Po zakończeniu treningu, utwórz zoptymalizowaną wersję modelu:

```bash
python freeze.py \
  --start_checkpoint=/tmp/polski_speech_model/conv.ckpt-4000 \
  --output_file=/tmp/frozen_graph.pb
```

**Dlaczego zamrażamy model:**
- Łączy wagi i architekturę w jeden plik
- Optymalizuje dla inferencji (nie dla treningu)
- Zmniejsza rozmiar
- Łatwiejszy do wdrożenia

## Jak działa rozpoznawanie mowy - Teoria

### 1. Przetwarzanie sygnału audio

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Fala        │    │  Spectro-    │    │  MFCC        │
│  dźwiękowa   │ -> │  gram        │ -> │  (cechy)     │
│  (surowe)    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Fala dźwiękowa:**
- Surowy sygnał z mikrofonu
- Wartości amplitudy w czasie
- Trudny do analizy bezpośrednio

**Spektrogram:**
- Reprezentacja częstotliwości w czasie
- Transformata Fouriera (STFT)
- Pokazuje jakie częstotliwości występują w danym momencie

**MFCC (Mel-Frequency Cepstral Coefficients):**
- Kompresja informacji ze spektrogramu
- Inspirowane ludzkim słuchem
- Standardowa reprezentacja dla mowy

### 2. Sieć neuronowa

Model używa konwolucyjnej sieci neuronowej (CNN):

```
┌─────────────┐
│  Wejście    │  Spektrogram 49x40
│  (MFCC)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Conv2D     │  Wykrywanie podstawowych wzorców
│  + ReLU     │  (krawędzie, prążki w spektrogramie)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Pooling    │  Redukcja rozmiaru
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Conv2D     │  Wykrywanie bardziej złożonych wzorców
│  + ReLU     │  (fonemy, sylaby)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Pooling    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Flatten    │  Spłaszczenie do wektora
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Dense      │  Klasyfikacja
│  (Softmax)  │  Prawdopodobieństwo dla każdego słowa
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Wynik      │  [0.89, 0.05, 0.03, 0.02]
│             │  "idz" z 89% pewnością
└─────────────┘
```

### 3. Rozpoznawanie w czasie rzeczywistym

Dla aplikacji robotycznych potrzebujemy rozpoznawania na bieżąco:

```python
from recognize_commands import RecognizeCommands

# Inicjalizacja
recognizer = RecognizeCommands(
    labels=['idz', 'stop', 'lewo', 'prawo'],
    average_window_duration_ms=1000,
    detection_threshold=0.7
)

# Pętla rozpoznawania
while robot.is_active():
    # Nagraj krótki fragment audio
    audio_data = microphone.record(duration_ms=100)
    
    # Przetwórz audio
    spectrogram = compute_spectrogram(audio_data)
    
    # Uruchom model
    predictions = model.predict(spectrogram)
    
    # Sprawdź czy wykryto komendę
    command = recognizer.process(predictions, current_time_ms)
    
    if command:
        print(f"Wykryto komendę: {command}")
        robot.execute_command(command)
```

## Integracja z robotem Unitree G1

### Kompletny przykład systemu sterowania głosem

```python
"""
System sterowania głosowego dla robota Unitree G1 EDU-U6
"""

import tensorflow as tf
import numpy as np
import pyaudio
import threading
from collections import deque

class VoiceControlledRobot:
    """Robot sterowany głosem używający TensorFlow."""
    
    def __init__(self, model_path, labels_path):
        """
        Inicjalizacja systemu.
        
        Args:
            model_path: Ścieżka do zamrożonego modelu .pb
            labels_path: Ścieżka do pliku z etykietami
        """
        # Ładowanie modelu TensorFlow
        self.graph = self.load_frozen_graph(model_path)
        self.labels = self.load_labels(labels_path)
        
        # Konfiguracja audio
        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION_MS = 100  # 100ms na chunk
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION_MS / 1000)
        
        # Buffer dla okna czasowego
        self.audio_buffer = deque(maxlen=16)  # 1.6s bufora
        
        # PyAudio do nagrywania
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Stan rozpoznawania
        self.is_listening = False
        self.last_command = None
        self.command_callback = None
        
    def load_frozen_graph(self, model_path):
        """Ładuje zamrożony model TensorFlow."""
        graph = tf.Graph()
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with graph.as_default():
            tf.import_graph_def(graph_def)
        
        return graph
    
    def load_labels(self, labels_path):
        """Wczytuje listę etykiet."""
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def start_listening(self, callback):
        """
        Rozpoczyna nasłuchiwanie poleceń głosowych.
        
        Args:
            callback: Funkcja wywoływana gdy rozpoznano komendę
                     callback(command_name, confidence)
        """
        self.command_callback = callback
        self.is_listening = True
        
        # Otwórz strumień audio
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        print("🎤 Nasłuchuję poleceń głosowych...")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback wywoływany dla każdego chunka audio."""
        # Konwertuj bajty do numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Dodaj do bufora
        self.audio_buffer.append(audio_data)
        
        # Gdy mamy wystarczająco danych, rozpoznaj
        if len(self.audio_buffer) == self.audio_buffer.maxlen:
            # Uruchom rozpoznawanie w osobnym wątku
            # (aby nie blokować strumienia audio)
            threading.Thread(
                target=self._recognize_from_buffer
            ).start()
        
        return (None, pyaudio.paContinue)
    
    def _recognize_from_buffer(self):
        """Rozpoznaje komendę z bufora audio."""
        # Połącz chunks w jeden sygnał
        audio_data = np.concatenate(list(self.audio_buffer))
        
        # Przetwórz audio do spektrogramu (uproszczone)
        # W prawdziwej implementacji użyj tf.signal lub librosa
        spectrogram = self._compute_spectrogram(audio_data)
        
        # Uruchom model
        with tf.compat.v1.Session(graph=self.graph) as sess:
            input_tensor = self.graph.get_tensor_by_name('import/input:0')
            output_tensor = self.graph.get_tensor_by_name('import/output:0')
            
            predictions = sess.run(output_tensor, {
                input_tensor: spectrogram
            })
        
        # Pobierz najlepszą predykcję
        predictions = np.squeeze(predictions)
        top_index = np.argmax(predictions)
        confidence = predictions[top_index]
        
        # Jeśli pewność wystarczająco wysoka
        if confidence > 0.7:
            command = self.labels[top_index]
            
            # Unikaj powtarzania tej samej komendy
            if command != self.last_command:
                self.last_command = command
                
                # Wywołaj callback
                if self.command_callback:
                    self.command_callback(command, confidence)
    
    def _compute_spectrogram(self, audio_data):
        """Oblicza spektrogram z sygnału audio."""
        # Tutaj powinno być pełne przetwarzanie audio
        # Dla uproszczenia zwracamy placeholder
        # W prawdziwej implementacji użyj:
        # - tf.signal.stft dla Short-Time Fourier Transform
        # - tf.signal.mfccs_from_log_mel_spectrogram dla MFCC
        
        # Przykład (uproszczony):
        stft = tf.signal.stft(
            tf.cast(audio_data, tf.float32),
            frame_length=480,
            frame_step=160
        )
        spectrogram = tf.abs(stft)
        
        # Reshape do oczekiwanego formatu
        spectrogram = tf.expand_dims(spectrogram, 0)
        spectrogram = tf.expand_dims(spectrogram, -1)
        
        return spectrogram.numpy()
    
    def stop_listening(self):
        """Zatrzymuje nasłuchiwanie."""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("🔇 Nasłuchiwanie zatrzymane.")


# Przykładowe użycie z robotem
def main():
    """Główna funkcja demonstracyjna."""
    
    # Inicjalizacja robota (przykład)
    # robot = UnitreeG1Robot()
    
    # Inicjalizacja systemu rozpoznawania głosu
    voice_system = VoiceControlledRobot(
        model_path='models/frozen_graph.pb',
        labels_path='models/labels.txt'
    )
    
    # Funkcja obsługująca komendy
    def handle_command(command, confidence):
        """Reaguje na rozpoznaną komendę."""
        print(f"\n🤖 Komenda: {command} (pewność: {confidence*100:.1f}%)")
        
        # Symulacja reakcji robota
        if command == 'idz':
            print("   → Robot idzie do przodu")
            # robot.move_forward()
        
        elif command == 'stop':
            print("   → Robot zatrzymuje się")
            # robot.stop()
        
        elif command == 'lewo':
            print("   → Robot obraca w lewo")
            # robot.turn_left()
        
        elif command == 'prawo':
            print("   → Robot obraca w prawo")
            # robot.turn_right()
        
        elif command == 'podnies':
            print("   → Robot podnosi obiekt")
            # robot.pick_up()
        
        elif command == 'poloz':
            print("   → Robot kładzie obiekt")
            # robot.put_down()
    
    # Rozpocznij nasłuchiwanie
    print("="*60)
    print(" System sterowania głosowego - Unitree G1")
    print("="*60)
    voice_system.start_listening(handle_command)
    
    try:
        # Czekaj na Ctrl+C
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nZatrzymywanie...")
        voice_system.stop_listening()
        print("Do widzenia!")


if __name__ == '__main__':
    main()
```

## Optymalizacja dla robota

### 1. Użycie TensorFlow Lite

Dla lepszej wydajności na robocie:

```bash
# Konwersja do TensorFlow Lite
python -c "
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    'frozen_graph.pb',
    input_arrays=['input'],
    output_arrays=['output']
)

# Optymalizacja
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Kwantyzacja (int8) dla jeszcze lepszej wydajności
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

### 2. Parametry optymalizacji

```python
# Kompromis między latencją a accuracy
OPTIMIZATION_PARAMS = {
    'window_size_ms': 30,        # Mniejsze okno = szybsza reakcja
    'window_stride_ms': 10,      # Mniejszy stride = lepsza accuracy
    'feature_bin_count': 32,     # Mniej cech = szybsze przetwarzanie
    'detection_threshold': 0.75  # Wyższy próg = mniej fałszywych wykryć
}
```

## Rozwiązywanie problemów

### Problem: Niska accuracy rozpoznawania

**Przyczyny i rozwiązania:**

1. **Za mało danych treningowych**
   - Rozwiązanie: Nagraj więcej przykładów (min. 100 na słowo)

2. **Szum tła**
   - Rozwiązanie: Dodaj przykłady z szumem do treningu
   - Rozwiązanie: Użyj filtrowania szumu (np. Wiener filter)

3. **Różni mówcy**
   - Rozwiązanie: Trenuj na nagraniach różnych osób
   - Rozwiązanie: Augmentacja danych (zmiana pitch, tempo)

### Problem: Fałszywe wykrycia

**Rozwiązanie:**
```python
# Zwiększ próg pewności
detection_threshold = 0.8  # Domyślnie 0.7

# Użyj uśredniania w czasie
average_window_duration_ms = 1500  # Domyślnie 1000

# Dodaj mechanizm potwierdzania
def confirm_command(command, history, min_repeats=2):
    """Potwierdza komendę tylko gdy pojawiła się wielokrotnie."""
    recent = history[-5:]
    count = recent.count(command)
    return count >= min_repeats
```

### Problem: Opóźnienie w rozpoznawaniu

**Rozwiązanie:**
```python
# Zmniejsz rozmiar okna
window_size_ms = 20  # Domyślnie 30

# Użyj mniejszego modelu
model_architecture = 'tiny_conv'  # Zamiast 'conv'

# Użyj TensorFlow Lite
# (3-5x szybsze niż pełny TensorFlow)
```

## Dodatkowe zasoby

### Dokumentacja
- [Oficjalny tutorial TensorFlow Audio](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Speech Commands Dataset](https://blog.research.google/2017/08/launching-speech-commands-dataset.html)

### Narzędzia
- [Audacity](https://www.audacityteam.org/) - Darmowy edytor audio
- [SoX](http://sox.sourceforge.net/) - Przetwarzanie audio z linii poleceń

### Zaawansowane systemy (dla ambitnych)
- [Kaldi](https://kaldi-asr.org/) - Profesjonalny system rozpoznawania mowy
- [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech) - Open source speech-to-text
- [Whisper](https://github.com/openai/whisper) - Model od OpenAI

## Podsumowanie

System rozpoznawania poleceń głosowych oferuje:
- ✅ Prosty interfejs głosowy dla robota
- ✅ Możliwość trenowania na własnych słowach (w tym polskich)
- ✅ Działanie w czasie rzeczywistym
- ✅ Odpowiednia wydajność dla robotyki

**Następne kroki:**
1. Przejdź przez przykład treningu
2. Nagraj własne polskie komendy
3. Wytrenuj model i przetestuj
4. Zintegruj z robotem Unitree G1
5. Eksperymentuj i udoskonalaj!

**Powodzenia w tworzeniu robota sterowanego głosem! 🤖🎤**
