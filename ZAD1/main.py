import numpy as np


def train_linear_neuron(N, M, num_runs=3, epochs=100, learning_rate=0.01):
    """
    Funkcja do trenowania neuronu liniowego i przeprowadzania eksperymentów.

    Argumenty:
    N (int): Liczba wag neuronu (wymiar wektora wejściowego).
    M (int): Liczba wzorców treningowych.
    num_runs (int): Liczba powtórzeń algorytmu z różnymi wagami początkowymi.
    epochs (int): Liczba epok treningowych.
    learning_rate (float): Współczynnik uczenia eta.
    """
    print(f"--- START EKSPERYMENTU: N = {N}, M = {M} ---")

    # 1. Generowanie losowego, ale STAŁEGO dla danego przypadku, zbioru treningowego
    # Ustawiamy ziarno losowości, aby zbiór treningowy był ten sam przy każdym uruchomieniu
    # tej funkcji, ale inny dla różnych par (N, M).
    np.random.seed(N + M)
    X = np.random.rand(M, N)  # Macierz wzorców wejściowych (M wierszy, N kolumn)
    # Generujemy "prawdziwe" wagi, aby stworzyć sensowne dane wyjściowe d
    true_w = np.random.randn(N)
    d = X @ true_w + np.random.randn(M) * 0.1  # Wyjścia = X * true_w + mały szum

    print(f"Wygenerowano zbiór treningowy o wymiarach X: {X.shape}, d: {d.shape}\n")

    final_weights = []

    # 2. Wykonanie algorytmu kilkukrotnie (num_runs)
    for run in range(num_runs):
        print(f"-> Uruchomienie {run + 1}/{num_runs}")
        # Losowe, RÓŻNIĄCE SIĘ w każdym wykonaniu, wartości wag początkowych
        w = np.random.randn(N)
        print(f"   Wagi początkowe: {np.round(w, 4)}")

        # Pętla treningowa
        for epoch in range(epochs):
            # Prezentacja każdego wzorca w epoce (uczenie online)
            for i in range(M):
                x_i = X[i]
                d_i = d[i]

                # Obliczenie odpowiedzi neuronu
                y_i = np.dot(w, x_i)

                # Obliczenie błędu
                error = d_i - y_i

                # Aktualizacja wag
                delta_w = learning_rate * error * x_i
                w = w + delta_w

        final_weights.append(w)
        print(f"   Wagi końcowe:    {np.round(w, 4)}")

        # Sprawdzenie błędu końcowego (Mean Squared Error)
        y_pred = X @ w
        mse = np.mean((d - y_pred) ** 2)
        print(f"   Końcowy błąd średniokwadratowy (MSE): {mse:.6f}\n")

    # 3. Porównanie wag końcowych
    print("-> Porównanie wag końcowych z różnych uruchomień:")
    for i, w in enumerate(final_weights):
        print(f"   Uruchomienie {i + 1}: {np.round(w, 4)}")

    # Sprawdzenie, czy wagi są do siebie zbliżone
    if len(final_weights) > 1:
        # Obliczamy odchylenie standardowe dla każdej wagi w poprzek uruchomień
        std_dev_of_weights = np.std(np.array(final_weights), axis=0)
        print(f"\nOdchylenie standardowe wag końcowych: {np.round(std_dev_of_weights, 4)}")
        if np.all(std_dev_of_weights < 1e-2):
            print("WNIOSKI: Wagi końcowe są **bardzo zbliżone** lub **identyczne**.")
        else:
            print("WNIOSKI: Wagi końcowe są **różne** w zależności od inicjalizacji.")

    print(f"--- KONIEC EKSPERYMENTU: N = {N}, M = {M} ---\n\n")


# Przeprowadzenie 3 eksperymentów
if __name__ == "__main__":
    # Przypadek 1: N < M (Układ nadokreślony)
    train_linear_neuron(N=3, M=10)

    # Przypadek 2: N = M (Układ oznaczony)
    train_linear_neuron(N=5, M=5)

    # Przypadek 3: N > M (Układ nieoznaczony)
    train_linear_neuron(N=10, M=4)