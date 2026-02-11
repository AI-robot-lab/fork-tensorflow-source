# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
=============================================================================
ROZPOZNAWANIE OBRAZÓW - Przewodnik dla studentów Politechniki Rzeszowskiej
=============================================================================

Ten skrypt demonstruje jak używać TensorFlow do rozpoznawania obiektów 
na obrazach. Jest to podstawowa aplikacja widzenia komputerowego, która 
może być wykorzystana w robocie Unitree G1 EDU-U6.

CO ROBI TEN PROGRAM:
1. Wczytuje przedtrenowany model sieci neuronowej (Inception V3)
2. Pobiera obraz do analizy
3. Przetwarza obraz (zmiana rozmiaru, normalizacja)
4. Klasyfikuje obraz - rozpoznaje co jest na obrazie
5. Wyświetla top 5 najbardziej prawdopodobnych obiektów

ZASTOSOWANIA W ROBOTYCE:
- Robot rozpoznaje obiekty w swoim otoczeniu
- Identyfikacja przedmiotów do manipulacji
- Wykrywanie ludzi i zwierząt
- Nawigacja oparta na rozpoznawaniu obiektów

UŻYCIE:
    python label_image.py --image=moj_obraz.jpg
    
WYMAGANE BIBLIOTEKI:
- TensorFlow: Framework uczenia maszynowego
- NumPy: Biblioteka do obliczeń numerycznych
- argparse: Parsowanie argumentów linii poleceń
=============================================================================
"""

import argparse  # Do przetwarzania argumentów z linii poleceń

import numpy as np  # Biblioteka do operacji na tablicach i macierzach
import tensorflow as tf  # Główna biblioteka TensorFlow

# Wyłączenie eager execution - używamy starszego API grafów
# DLACZEGO: Ten przykład używa starszego stylu TensorFlow z grafami obliczeniowymi
tf.compat.v1.disable_eager_execution()

def load_graph(model_file):
  """
  Ładuje przedtrenowany model TensorFlow z pliku.
  
  KROK 1: ŁADOWANIE MODELU
  -------------------------
  Model sieci neuronowej jest zapisany w formacie Protocol Buffer (.pb).
  Zawiera on strukturę sieci (warstwy) oraz wytrenowane wagi.
  
  CO TO JEST GRAF (Graph):
  - Graf to reprezentacja obliczeń w TensorFlow
  - Składa się z węzłów (operacje) i krawędzi (tensory - dane)
  - Pozwala na efektywne wykonywanie obliczeń
  
  DLACZEGO ŁADUJEMY MODEL Z PLIKU:
  - Trening modelu zajmuje bardzo dużo czasu (dni/tygodnie)
  - Model Inception V3 był trenowany na milionach obrazów
  - Możemy użyć gotowego modelu zamiast trenować od zera
  
  Args:
      model_file (str): Ścieżka do pliku .pb z zamrożonym modelem
  
  Returns:
      tf.Graph: Załadowany graf obliczeniowy TensorFlow
  
  PRZYKŁAD UŻYCIA DLA ROBOTA:
      graph = load_graph("models/inception_v3.pb")
      # Teraz robot może używać tego modelu do rozpoznawania obiektów
  """
  # Tworzymy nowy, pusty graf
  graph = tf.Graph()
  
  # GraphDef to definicja grafu w formacie Protocol Buffer
  graph_def = tf.compat.v1.GraphDef()

  # Otwieramy plik modelu w trybie binarnym ('rb' = read binary)
  with open(model_file, "rb") as f:
    # ParseFromString wczytuje dane binarne do graph_def
    graph_def.ParseFromString(f.read())
  
  # Ustawiamy graph jako domyślny kontekst
  with graph.as_default():
    # Importujemy definicję grafu do naszego obiektu graph
    # Od teraz graph zawiera całą strukturę sieci neuronowej
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  """
  Wczytuje i przetwarza obraz do formatu wymaganego przez sieć neuronową.
  
  KROK 2: PRZETWARZANIE OBRAZU (PREPROCESSING)
  ---------------------------------------------
  Sieci neuronowe wymagają obrazów w bardzo konkretnym formacie:
  - Określony rozmiar (299x299 dla Inception V3)
  - Znormalizowane wartości pikseli
  - Odpowiednia liczba kanałów kolorów (RGB = 3)
  
  DLACZEGO TO JEST WAŻNE:
  - Model był trenowany na obrazach 299x299, więc oczekuje takiego rozmiaru
  - Normalizacja (dzielenie przez 255) zamienia wartości pikseli z zakresu 
    [0, 255] na zakres [0, 1], co ułatwia uczenie sieci
  - Spójny format zapewnia poprawne działanie modelu
  
  PROCES KROK PO KROKU:
  1. Wczytanie pliku obrazu
  2. Dekodowanie obrazu (różne formaty: JPEG, PNG, GIF, BMP)
  3. Konwersja do liczb zmiennoprzecinkowych (float32)
  4. Dodanie wymiaru batch (model może przetwarzać wiele obrazów naraz)
  5. Zmiana rozmiaru do 299x299 pikseli
  6. Normalizacja wartości pikseli
  
  Args:
      file_name (str): Ścieżka do pliku obrazu
      input_height (int): Wymagana wysokość obrazu (domyślnie 299)
      input_width (int): Wymagana szerokość obrazu (domyślnie 299)
      input_mean (int): Średnia do normalizacji (domyślnie 0)
      input_std (int): Odchylenie standardowe do normalizacji (domyślnie 255)
  
  Returns:
      numpy.ndarray: Przetworzony obraz gotowy do podania do sieci
  
  PRZYKŁAD UŻYCIA DLA ROBOTA:
      # Robot zrobił zdjęcie kamerą
      image_tensor = read_tensor_from_image_file("robot_camera.jpg")
      # Teraz może przeanalizować co widzi
  """
  input_name = "file_reader"
  output_name = "normalized"
  
  # KROK 1: Wczytanie pliku obrazu
  # tf.io.read_file wczytuje plik jako surowe bajty
  file_reader = tf.io.read_file(file_name, input_name)
  
  # KROK 2: Dekodowanie obrazu w zależności od formatu pliku
  # Różne formaty obrazów wymagają różnych dekoderów
  if file_name.endswith(".png"):
    # PNG: Format bezstratny, często używany dla grafiki
    image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    # GIF: Może zawierać animacje, bierzemy pierwszą klatkę
    image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    # BMP: Stary format Microsoftu, używany rzadko
    image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
  else:
    # JPEG: Najpopularniejszy format dla zdjęć
    # Domyślnie zakładamy JPEG jeśli nie rozpoznano rozszerzenia
    image_reader = tf.io.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  
  # KROK 3: Konwersja do liczb zmiennoprzecinkowych
  # Obrazy są zapisane jako liczby całkowite (0-255)
  # Sieci neuronowe lepiej działają na liczbach zmiennoprzecinkowych
  float_caster = tf.cast(image_reader, tf.float32)
  
  # KROK 4: Dodanie wymiaru batch
  # Model oczekuje kształtu [batch_size, height, width, channels]
  # expand_dims dodaje wymiar batch=1 (przetwarzamy jeden obraz)
  # PRZED: [299, 299, 3]  <- wysokość, szerokość, kolory
  # PO:    [1, 299, 299, 3] <- batch, wysokość, szerokość, kolory
  dims_expander = tf.expand_dims(float_caster, 0)
  
  # KROK 5: Zmiana rozmiaru obrazu
  # resize_bilinear używa interpolacji dwuliniowej dla gładkiego skalowania
  # Zmienia rozmiar na dokładnie [input_height, input_width]
  resized = tf.compat.v1.image.resize_bilinear(
      dims_expander, [input_height, input_width]
  )
  
  # KROK 6: Normalizacja
  # normalized = (pixel_value - mean) / std
  # Dla standardowej normalizacji: (pixel - 0) / 255
  # To przekształca zakres z [0, 255] na [0, 1]
  # DLACZEGO: Sieci neuronowe lepiej uczą się na znormalizowanych danych
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  
  # KROK 7: Wykonanie obliczeń
  # W TensorFlow 1.x musimy utworzyć sesję aby wykonać operacje
  sess = tf.compat.v1.Session()
  
  # sess.run wykonuje wszystkie operacje i zwraca końcowy wynik
  return sess.run(normalized)


def load_labels(label_file):
  """
  Wczytuje plik z etykietami (nazwami kategorii).
  
  KROK 3: ŁADOWANIE ETYKIET
  --------------------------
  Model zwraca tylko liczby (prawdopodobieństwa dla każdej kategorii).
  Potrzebujemy pliku z nazwami, aby wiedzieć co oznaczają te liczby.
  
  Format pliku etykiet:
  - Każda linia = jedna kategoria
  - Linia 0 = kategoria 0, linia 1 = kategoria 1, itd.
  - Dla ImageNet: 1000 linii (1000 kategorii obiektów)
  
  Przykład zawartości:
      tench              (ryba lin)
      goldfish           (złota rybka)
      great white shark  (wielki biały rekin)
      ...
      military uniform   (mundur wojskowy)
      ...
  
  Args:
      label_file (str): Ścieżka do pliku tekstowego z etykietami
  
  Returns:
      list: Lista nazw kategorii (stringów)
  
  PRZYKŁAD UŻYCIA DLA ROBOTA:
      labels = load_labels("imagenet_labels.txt")
      # Jeśli model zwróci indeks 653, robot wie że to labels[653]
      print(labels[653])  # Wypisze: "military uniform"
  """
  # GFile to wrapper TensorFlow do operacji na plikach
  # Działa zarówno lokalnie jak i w chmurze (Google Cloud Storage)
  proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
  
  # Usuwamy białe znaki z końca każdej linii (.rstrip())
  # List comprehension tworzy listę przetworzonych etykiet
  return [l.rstrip() for l in proto_as_ascii_lines]


if __name__ == "__main__":
  """
  GŁÓWNA CZĘŚĆ PROGRAMU
  =====================
  
  To jest punkt startowy programu. Wykonuje się gdy uruchamiasz:
      python label_image.py
  
  PRZEPŁYW WYKONANIA:
  1. Ustawienie domyślnych parametrów
  2. Parsowanie argumentów z linii poleceń
  3. Załadowanie modelu
  4. Przetworzenie obrazu
  5. Uruchomienie klasyfikacji
  6. Wyświetlenie wyników
  """
  
  # ========================================================================
  # KROK 1: KONFIGURACJA DOMYŚLNYCH PARAMETRÓW
  # ========================================================================
  # Te wartości są używane jeśli nie podasz własnych argumentów
  
  # Domyślny obraz testowy - zdjęcie Grace Hopper (pionierka informatyki)
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  
  # Ścieżka do przedtrenowanego modelu Inception V3
  # .pb = Protocol Buffer, format do zapisywania modeli TensorFlow
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  
  # Plik z 1000 etykiet z datasetu ImageNet
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  
  # Parametry przetwarzania obrazu
  # UWAGA: Te wartości muszą pasować do tego jak model był trenowany!
  input_height = 299  # Wysokość w pikselach (Inception V3 wymaga 299)
  input_width = 299   # Szerokość w pikselach
  input_mean = 0      # Średnia dla normalizacji
  input_std = 255     # Odchylenie standardowe dla normalizacji
  
  # Nazwy warstw w modelu
  # input_layer: warstwa przyjmująca dane wejściowe
  # output_layer: warstwa zwracająca predykcje
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  # ========================================================================
  # KROK 2: PARSOWANIE ARGUMENTÓW Z LINII POLECEŃ
  # ========================================================================
  # Użytkownik może nadpisać domyślne wartości podając argumenty
  # Przykład: python label_image.py --image=moje_zdjecie.jpg
  
  parser = argparse.ArgumentParser()
  
  # Definiujemy możliwe argumenty
  # help: opis wyświetlany gdy użytkownik wpisze --help
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  
  # Parsujemy argumenty dostarczone przez użytkownika
  args = parser.parse_args()

  # Nadpisujemy domyślne wartości jeśli użytkownik podał własne
  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  # ========================================================================
  # KROK 3: ZAŁADOWANIE MODELU I PRZETWORZENIE OBRAZU
  # ========================================================================
  
  # Ładujemy model do pamięci
  # To może chwilę potrwać dla dużych modeli
  print("Ładowanie modelu...")
  graph = load_graph(model_file)
  print("Model załadowany!")
  
  # Przetwarzamy obraz do formatu wymaganego przez model
  print(f"Przetwarzanie obrazu: {file_name}")
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
  print("Obraz przetworzony!")

  # ========================================================================
  # KROK 4: URUCHOMIENIE MODELU (INFERENCE)
  # ========================================================================
  
  # Tworzymy pełne nazwy warstw (prefiks "import/" jest dodawany automatycznie)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  
  # Pobieramy operacje (węzły grafu) po nazwie
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  # Uruchamiamy model w sesji TensorFlow
  print("Uruchamianie klasyfikacji...")
  with tf.compat.v1.Session(graph=graph) as sess:
    # sess.run wykonuje obliczenia
    # Podajemy dane wejściowe (nasz obraz) i otrzymujemy wyniki
    # results będzie tablicą 1000 liczb - prawdopodobieństwo dla każdej kategorii
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  
  # ========================================================================
  # KROK 5: PRZETWORZENIE I WYŚWIETLENIE WYNIKÓW
  # ========================================================================
  
  # squeeze usuwa wymiary o rozmiarze 1
  # PRZED: [1, 1000] <- batch=1, klasy=1000
  # PO:    [1000]    <- tylko klasy
  results = np.squeeze(results)

  # argsort zwraca indeksy posortowane według wartości
  # [-5:] bierze ostatnie 5 (największe wartości)
  # [::-1] odwraca kolejność (od największego do najmniejszego)
  # Rezultat: indeksy 5 najbardziej prawdopodobnych kategorii
  top_k = results.argsort()[-5:][::-1]
  
  # Ładujemy nazwy kategorii
  labels = load_labels(label_file)
  
  # Wyświetlamy wyniki
  print("\n" + "="*60)
  print("WYNIKI ROZPOZNAWANIA:")
  print("="*60)
  for i in top_k:
    # Dla każdej z top 5 kategorii wyświetlamy:
    # - Nazwę kategorii (labels[i])
    # - Prawdopodobieństwo (results[i] to wartość od 0 do 1)
    print(f"{labels[i]:30s} {results[i]*100:6.2f}%")
  print("="*60)
  
  """
  PRZYKŁADOWY WYNIK:
  ============================================================
  WYNIKI ROZPOZNAWANIA:
  ============================================================
  military uniform                83.43%
  mortarboard                      2.19%
  academic gown                    1.04%
  pickelhaube                      0.80%
  bulletproof vest                 0.54%
  ============================================================
  
  INTERPRETACJA DLA ROBOTA:
  - Robot widzi osobę w mundurze wojskowym z 83.43% pewnością
  - Robot jest dość pewny swojej odpowiedzi (>80%)
  - Robot może teraz zareagować odpowiednio (np. powitać oficera)
  """
