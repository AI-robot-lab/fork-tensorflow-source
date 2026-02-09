# TensorFlow - Przewodnik dla Studentów Politechniki Rzeszowskiej

<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

## Wprowadzenie

TensorFlow to otwarta platforma do uczenia maszynowego (machine learning), stworzona przez Google. W ramach projektu z robotem humanoidalnym **Unitree G1 EDU-U6** będziemy wykorzystywać TensorFlow do implementacji inteligentnych funkcji przetwarzania obrazu, rozpoznawania mowy i podejmowania decyzji przez robota.

## Dlaczego TensorFlow w robotyce?

TensorFlow umożliwia robotowi:
- **Widzenie komputerowe** - rozpoznawanie obiektów, twarzy, gestów
- **Przetwarzanie mowy** - rozumienie poleceń głosowych
- **Podejmowanie decyzji** - uczenie się na podstawie danych i doświadczeń
- **Nawigację** - mapowanie przestrzeni i unikanie przeszkód

## Co to jest TensorFlow?

TensorFlow to kompleksowa platforma open source do uczenia maszynowego, która oferuje:
- Elastyczny ekosystem [narzędzi](https://www.tensorflow.org/resources/tools)
- Obszerne [biblioteki](https://www.tensorflow.org/resources/libraries-extensions)
- Aktywną [społeczność](https://www.tensorflow.org/community) programistów i badaczy

Pierwotnie opracowany przez zespół Google Brain do badań nad uczeniem maszynowym i sieciami neuronowymi, TensorFlow jest obecnie używany w wielu różnych dziedzinach, w tym w robotyce.

## Instalacja

### Podstawowa instalacja z obsługą GPU

```bash
# Instalacja TensorFlow z obsługą kart graficznych CUDA
$ pip install tensorflow
```

### Wersja tylko dla CPU (bez GPU)

```bash
# Lżejsza wersja tylko dla procesora CPU
$ pip install tensorflow-cpu
```

### Aktualizacja do najnowszej wersji

```bash
# Dodaj flagę --upgrade do aktualizacji
$ pip install tensorflow --upgrade
```

**Uwaga**: Aby korzystać z GPU, potrzebujesz karty graficznej zgodnej z CUDA. Więcej informacji w [przewodniku instalacji GPU](https://www.tensorflow.org/install/gpu).

## Twój pierwszy program w TensorFlow

```bash
$ python
```

```python
>>> import tensorflow as tf
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('Witaj, TensorFlow!')
>>> hello.numpy()
b'Witaj, TensorFlow!'
```

## Struktura tego repozytorium

### Główne katalogi:

- **tensorflow/examples/** - Przykłady demonstrujące różne funkcje TensorFlow
  - `label_image/` - Rozpoznawanie obiektów na obrazach (klasyfikacja)
  - `speech_commands/` - Rozpoznawanie poleceń głosowych
  - `adding_an_op/` - Tworzenie własnych operacji TensorFlow
  
- **tensorflow/lite/** - TensorFlow Lite dla urządzeń mobilnych i embedded (w tym robotów!)
- **tensorflow/python/** - API Pythona dla TensorFlow
- **tensorflow/core/** - Rdzeń TensorFlow napisany w C++

## Kluczowe przykłady dla projektu z robotem

### 1. Rozpoznawanie obrazów (label_image)

Wykorzystanie: Robot może rozpoznawać obiekty, ludzi, przeszkody.

```bash
cd tensorflow/examples/label_image
python label_image.py --image=obraz_do_rozpoznania.jpg
```

### 2. Rozpoznawanie poleceń głosowych (speech_commands)

Wykorzystanie: Robot reaguje na polecenia głosowe.

```bash
cd tensorflow/examples/speech_commands
python train.py
```

## Zasoby edukacyjne

### Oficjalna dokumentacja
- [Samouczki TensorFlow](https://www.tensorflow.org/tutorials/) - Przewodniki krok po kroku
- [Dokumentacja API](https://www.tensorflow.org/api_docs/) - Szczegółowa referencja funkcji

### Kursy online (w języku angielskim)
- [Coursera - TensorFlow](https://www.coursera.org/search?query=TensorFlow)
- [Udacity - TensorFlow](https://www.udacity.com/courses/all?search=TensorFlow)
- [Edx - TensorFlow](https://www.edx.org/search?q=TensorFlow)

### Społeczność
- [Forum TensorFlow](https://discuss.tensorflow.org/) - Pytania i dyskusje
- [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow) - Pomoc techniczna
- [Blog TensorFlow](https://blog.tensorflow.org) - Nowości i poradniki

## Aplikacje w robotyce - Unitree G1 EDU-U6

Robot humanoidalny Unitree G1 EDU-U6 może wykorzystywać TensorFlow do:

1. **Widzenia komputerowego**
   - Rozpoznawanie obiektów w otoczeniu
   - Śledzenie twarzy i gestów ludzi
   - Wykrywanie przeszkód

2. **Przetwarzania mowy**
   - Rozpoznawanie poleceń głosowych
   - Interakcja człowiek-robot
   - Rozumienie intencji użytkownika

3. **Manipulacji obiektami**
   - Identyfikacja przedmiotów do chwytania
   - Planowanie trajektorii ruchu
   - Kontrola siły chwytu

4. **Nawigacji**
   - Mapowanie otoczenia
   - Planowanie ścieżki
   - Unikanie kolizji

## Dalsze kroki

1. Zapoznaj się z [przewodnikiem dla robota Unitree G1](UNITREE_G1_GUIDE_PL.md)
2. Przejdź przez przykłady w katalogu `tensorflow/examples/`
3. Przeczytaj komentarze w kodzie - są szczegółowe i w języku polskim
4. Eksperymentuj z własnymi danymi i modelami

## Wytyczne dla współtwórców

Jeśli chcesz wnieść wkład do TensorFlow, zapoznaj się z:
- [Wytycznymi dla współtwórców](CONTRIBUTING.md)
- [Kodeksem postępowania](CODE_OF_CONDUCT.md)

**Śledzimy zgłoszenia przez [GitHub Issues](https://github.com/tensorflow/tensorflow/issues)**

## Licencja

[Apache License 2.0](LICENSE)

---

**Powodzenia w nauce TensorFlow i pracy z robotem Unitree G1 EDU-U6!** 🤖🎓
