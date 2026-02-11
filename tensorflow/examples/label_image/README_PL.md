# Rozpoznawanie Obrazów - Demo TensorFlow (C++ i Python)

## Wprowadzenie dla Studentów

Ten przykład pokazuje jak wykorzystać przedtrenowaną sieć neuronową TensorFlow do rozpoznawania obiektów na obrazach. Jest to fundamentalna technika widzenia komputerowego, którą można zastosować w robocie Unitree G1 EDU-U6.

## Opis

### Co robi ten program?

Demo używa modelu Google Inception do klasyfikacji plików obrazów przekazanych jako argumenty linii poleceń.

**Model Inception V3:**
- Wytrenowany na 1,000 kategorii obiektów z konkursu ImageNet
- Jeden z najlepszych modeli do rozpoznawania obiektów
- Może rozpoznać: zwierzęta, przedmioty, pojazdy, rośliny i wiele więcej

### Zastosowania w robotyce (Unitree G1 EDU-U6)

Robot może wykorzystać ten kod do:
- **Rozpoznawania obiektów** - identyfikacja przedmiotów do podniesienia
- **Wykrywania ludzi** - rozpoznawanie osób w otoczeniu
- **Nawigacji** - identyfikacja przeszkód i punktów orientacyjnych
- **Interakcji** - reagowanie na pokazywane obiekty

## Instalacja i uruchomienie

### Krok 1: Pobranie modelu

Model TensorFlow `GraphDef` zawierający definicję modelu i wagi nie jest zawarty w repozytorium ze względu na rozmiar. Musisz najpierw pobrać plik do katalogu `data`:

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C tensorflow/examples/label_image/data -xz
```

**Co się dzieje w tym poleceniu:**
- `curl -L` - pobiera plik z internetu
- `tar -xz` - rozpakowuje archiwum .tar.gz
- `-C tensorflow/...` - rozpakowanie do konkretnego katalogu

### Krok 2: Sprawdzenie etykiet

Po rozpakowaniu, zobacz plik z etykietami w katalogu data. Zawiera on 1,000 możliwych kategorii używanych w konkursie Imagenet.

```bash
$ cat tensorflow/examples/label_image/data/imagenet_slim_labels.txt | head -10
```

Przykładowe kategorie:
- tench (ryba lin)
- goldfish (złota rybka)
- great white shark (wielki biały rekin)
- tiger shark (rekin tygrysowy)
- ...

### Krok 3: Kompilacja (opcjonalnie dla wersji C++)

Jeśli udało Ci się zbudować główny framework TensorFlow, powinieneś mieć wszystko co potrzebne do uruchomienia tego przykładu.

Aby zbudować wersję C++, uruchom:

```bash
$ bazel build tensorflow/examples/label_image/...
```

**Co robi bazel:**
- Kompiluje kod C++
- Linkuje z bibliotekami TensorFlow
- Tworzy wykonywalny program binarny

### Krok 4: Uruchomienie

#### Wersja C++

Uruchom skompilowany program:

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image
```

To użyje domyślnego przykładowego obrazu dostarczonego z frameworkiem i powinno wyświetlić wynik podobny do:

```
I tensorflow/examples/label_image/main.cc:206] military uniform (653): 0.834306
I tensorflow/examples/label_image/main.cc:206] mortarboard (668): 0.0218692
I tensorflow/examples/label_image/main.cc:206] academic gown (401): 0.0103579
I tensorflow/examples/label_image/main.cc:206] pickelhaube (716): 0.00800814
I tensorflow/examples/label_image/main.cc:206] bulletproof vest (466): 0.00535088
```

**Interpretacja wyników:**
- W tym przypadku używamy domyślnego obrazu Admiral Grace Hopper
- Sieć poprawnie rozpoznaje że jest ubrana w mundur wojskowy (military uniform)
- Wysoki wynik 0.83 (83%) oznacza wysoką pewność rozpoznania

#### Testowanie na własnych obrazach

Wypróbuj na własnych obrazach dodając argument --image:

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image --image=moj_obraz.png
```

**Wskazówki dla najlepszych wyników:**
- Używaj obrazów JPEG lub PNG
- Obraz powinien wyraźnie pokazywać obiekt
- Dobrze oświetlone zdjęcia działają lepiej
- Obiekt powinien być głównym elementem obrazu

#### Wersja Python

`label_image.py` to implementacja w Pythonie odpowiadająca kodowi C++. Daje bardziej intuicyjne mapowanie między C++ a Pythonem niż kod Pythona wspomniany w [samouczku Inception](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/images/image_recognition.md) i może być łatwiejsza do dodania wizualizacji lub kodu debugującego.

**Kompilacja z bazel:**

```bash
$ bazel build tensorflow/examples/label_image/...
```

Po kompilacji uruchom:

```bash
$ bazel-bin/tensorflow/examples/label_image/label_image_py
```

**Lub bezpośrednio z Pythonem:**

Jeśli masz zainstalowany pakiet tensorflow python, możesz uruchomić bezpośrednio:

```bash
$ python3 tensorflow/examples/label_image/label_image.py
```

Otrzymasz wynik podobny do:

```
Ładowanie modelu...
Model załadowany!
Przetwarzanie obrazu: tensorflow/examples/label_image/data/grace_hopper.jpg
Obraz przetworzony!
Uruchamianie klasyfikacji...

============================================================
WYNIKI ROZPOZNAWANIA:
============================================================
military uniform                83.43%
mortarboard                      2.19%
academic gown                    1.04%
pickelhaube                      0.80%
bulletproof vest                 0.54%
============================================================
```

## Jak działa kod - Krok po kroku

### 1. Ładowanie modelu (`load_graph`)

```python
graph = load_graph(model_file)
```

- Wczytuje przedtrenowany model z pliku .pb
- Model zawiera strukturę sieci neuronowej i wytrenowane wagi
- Trening tego modelu zajął wiele dni na potężnych komputerach

### 2. Przetwarzanie obrazu (`read_tensor_from_image_file`)

```python
t = read_tensor_from_image_file(file_name, ...)
```

Kroki przetwarzania:
1. **Wczytanie** - Odczytanie pliku obrazu
2. **Dekodowanie** - Konwersja JPEG/PNG do pikseli
3. **Zmiana rozmiaru** - Skalowanie do 299x299 pikseli
4. **Normalizacja** - Przekształcenie wartości pikseli do zakresu [0, 1]

**Dlaczego 299x299?**
- Model Inception V3 został wytrenowany na obrazach tego rozmiaru
- Musi otrzymać dane w takim samym formacie jakiego używał podczas treningu

### 3. Uruchomienie klasyfikacji

```python
results = sess.run(output_operation.outputs[0], {...})
```

- Sieć neuronowa przetwarza obraz przez wiele warstw
- Każda warstwa wyodrębnia różne cechy (krawędzie, tekstury, kształty)
- Ostatnia warstwa zwraca prawdopodobieństwa dla 1000 kategorii

### 4. Wyświetlenie wyników

```python
top_k = results.argsort()[-5:][::-1]
```

- Sortujemy wyniki aby znaleźć top 5
- Wyświetlamy nazwy kategorii z prawdopodobieństwami

## Integracja z robotem Unitree G1 EDU-U6

### Przykładowa implementacja

```python
import robot_sdk  # Hipotetyczne SDK robota

# Inicjalizacja
robot = robot_sdk.connect()
graph = load_graph("inception_v3.pb")

# Pętla główna robota
while robot.is_active():
    # Pobierz obraz z kamery robota
    image = robot.camera.capture()
    
    # Zapisz tymczasowo
    image.save("temp_image.jpg")
    
    # Rozpoznaj obiekt
    tensor = read_tensor_from_image_file("temp_image.jpg")
    results = classify_image(graph, tensor)
    
    # Pobierz najlepszy wynik
    top_category = results[0]
    
    # Robot reaguje na rozpoznany obiekt
    if top_category == "ball" and confidence > 0.8:
        robot.say("Widzę piłkę!")
        robot.move_to_object()
    
    elif top_category == "person":
        robot.say("Witam!")
        robot.wave_hand()
```

### Możliwe rozszerzenia dla robota

1. **Ciągłe monitorowanie**
   - Robot nieustannie analizuje obraz z kamery
   - Reaguje na pojawiające się obiekty

2. **Śledzenie obiektów**
   - Robot śledzi ruch rozpoznanego obiektu
   - Obraca się aby utrzymać obiekt w polu widzenia

3. **Manipulacja obiektami**
   - Robot rozpoznaje obiekt
   - Planuje jak go chwycić
   - Wykonuje manipulację

4. **Interakcja z ludźmi**
   - Rozpoznaje czy w pobliżu jest osoba
   - Reaguje odpowiednio (wita, podąża, unika)

## Dodatkowe informacje

### Szczegółowy przewodnik

Aby uzyskać bardziej szczegółowe spojrzenie na ten kod, sprawdź sekcję C++ w [samouczku Inception](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/images/image_recognition.md).

### Inne przykłady i języki

- **Java**: Zobacz [Java README](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java)
- **Go**: Zobacz [godoc example](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go#ex-package)
- **TensorFlow Lite**: Dla urządzeń mobilnych i embedded (idealny dla robotów!)

## Rozwiązywanie problemów

### Problem: "Model nie znaleziony"

**Rozwiązanie:**
```bash
# Sprawdź czy model został pobrany
ls tensorflow/examples/label_image/data/*.pb

# Jeśli nie ma, pobierz ponownie
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C tensorflow/examples/label_image/data -xz
```

### Problem: "Słabe wyniki rozpoznawania"

**Rozwiązanie:**
- Użyj lepszej jakości obrazów
- Upewnij się że obiekt jest wyraźnie widoczny
- Wypróbuj różne modele (np. MobileNet, ResNet)
- Dostosuj parametry normalizacji

### Problem: "Program działa zbyt wolno"

**Rozwiązanie:**
- Użyj GPU zamiast CPU
- Użyj TensorFlow Lite dla szybszej inferencji
- Zmniejsz rozdzielczość obrazu
- Użyj mniejszego modelu

## Podsumowanie

Ten przykład demonstruje:
- ✅ Ładowanie przedtrenowanych modeli TensorFlow
- ✅ Przetwarzanie obrazów dla sieci neuronowych
- ✅ Klasyfikację obrazów (rozpoznawanie obiektów)
- ✅ Interpretację wyników sieci neuronowej

**Następne kroki:**
1. Uruchom przykład na różnych obrazach
2. Zrozum każdą linię kodu (przeczytaj komentarze!)
3. Eksperymentuj z własnymi modyfikacjami
4. Zintegruj z projektem robota Unitree G1

**Powodzenia! 🤖📸**
