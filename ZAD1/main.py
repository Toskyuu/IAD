import numpy as np
import os


def train_and_analyze(N, M, num_runs=3, epochs=100, eta=0.01):
    """
    Implementuje i analizuje trening neuronu liniowego dla zadanego N i M,
    używając danych treningowych z pliku TXT.
    """
    filename = f"dane_N{N}_M{M}.txt"

    if os.path.exists(filename):
        print(f"Znaleziono plik '{filename}'. Wczytuję dane treningowe...")
        # Wczytujemy dane. Ostatnia kolumna to 'z', reszta to 'X'.
        data = np.loadtxt(filename)
        X = data[:, :-1]
        z = data[:, -1]
    else:
        print(f"Nie znaleziono pliku '{filename}'. Generuję nowe dane treningowe...")
        # Używamy N+M jako ziarna losowości, aby dane były powtarzalne
        np.random.seed(N + M)

        X = np.random.rand(M, N) * 2 - 1
        prawdziwe_wagi = np.random.randn(N)
        z = X @ prawdziwe_wagi + np.random.randn(M) * 0.1

        data_to_save = np.hstack((X, z.reshape(-1, 1)))
        np.savetxt(filename, data_to_save, fmt="%.8f")
        print(f"Dane zostały zapisane w pliku '{filename}'.")

    print(f"\n--- START EKSPERYMENTU: N = {N}, M = {M} ---")
    print(f"Zbiór treningowy: X ma wymiar {X.shape}, z ma wymiar {z.shape}\n")

    final_weights_list = []

    for run in range(num_runs):
        print(f"-> PRZEBIEG {run + 1}/{num_runs}")

        w = np.random.uniform(low=-0.5, high=0.5, size=N)
        print(f"   Wagi początkowe (w): {np.round(w, 4)}")

        # Pętla epok
        for k in range(epochs):
            # Pętla po wzorcach
            for mi in range(M):
                x_mi = X[mi]
                z_mi = z[mi]
                y = np.dot(w, x_mi)
                error = z_mi - y
                delta_w = eta * error * x_mi
                w = w + delta_w

        final_weights_list.append(w)
        print(f"   Wagi końcowe (w):    {np.round(w, 4)}")

        y_final = X @ w
        mse = np.mean((z - y_final) ** 2)
        print(f"   Końcowy błąd (MSE): {mse:.6f}\n")

    print("-> Porównanie i analiza wag końcowych:")

    std_dev_of_weights = np.std(np.array(final_weights_list), axis=0)
    print(f"   Odchylenie standardowe wag: {np.round(std_dev_of_weights, 4)}")

    if np.all(std_dev_of_weights < 1e-2):
        print("   WNIOSKI: Wagi końcowe są **zbieżne** (praktycznie identyczne) w każdym przebiegu.")
    else:
        print("   WNIOSKI: Wagi końcowe są **różne** w zależności od losowej inicjalizacji.")

    print(f"--- KONIEC EKSPERYMENTU: N = {N}, M = {M} ---\n\n")


# Główna część programu
if __name__ == "__main__":
    # PRZYPADEK 1: N < M
    train_and_analyze(N=3, M=10)

    # PRZYPADEK 2: N = M
    train_and_analyze(N=5, M=5)

    # PRZYPADEK 3: N > M
    train_and_analyze(N=10, M=4)