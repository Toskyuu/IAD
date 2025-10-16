import numpy as np
import os


def train_and_analyze(N, M, num_runs=3, epochs=100, eta=0.05):
    filename = f"dane_N{N}_M{M}.txt"

    if os.path.exists(filename):
        print(f"Znaleziono plik '{filename}'. Wczytuję dane treningowe...")
        data = np.loadtxt(filename)
        X = data[:, :-1]
        z = data[:, -1]
    else:
        print(f"Nie znaleziono pliku '{filename}'. Generuję nowe dane treningowe...")
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
    mse_list = []

    for run in range(num_runs):
        print(f"-> PRZEBIEG {run + 1}/{num_runs}")

        w = np.random.uniform(low=-0.5, high=0.5, size=N)
        print(f"   Wagi początkowe (w): {np.round(w, 4)}")
        Y = []

        for k in range(epochs):
            for mi in range(M):
                x_mi = X[mi]
                z_mi = z[mi]
                y = np.dot(w, x_mi)
                error = z_mi - y
                delta_w = eta * error * x_mi
                w = w + delta_w

        final_weights_list.append(w)
        y_final = X @ w
        mse = np.mean((z - y_final) ** 2)
        mse_list.append(mse)
        print("Szukane wartości: ", z)
        print("Znalezione wartości: ", y_final)

        print(f"   Wagi końcowe (w):    {np.round(w, 4)}")
        print(f"   Końcowy błąd (MSE): {mse:.6f}\n")

    print("-> Porównanie i analiza wag końcowych:")
    std_dev_of_weights = np.std(np.array(final_weights_list), axis=0)
    mean_std_dev = np.mean(std_dev_of_weights)

    print(f"   Odchylenie standardowe wag: {np.round(std_dev_of_weights, 4)}")
    print(f"   Średnie odchylenie wag: {mean_std_dev:.6f}")
    print(f"   Średni błąd MSE: {np.mean(mse_list):.6f}")



    print(f"--- KONIEC EKSPERYMENTU: N = {N}, M = {M} ---\n\n")


if __name__ == "__main__":
    # PRZYPADEK 1: N < M
    train_and_analyze(N=3, M=5)

    # # PRZYPADEK 2: N = M
    # train_and_analyze(N=3, M=10)
    #
    # # PRZYPADEK 3: N > M
    # train_and_analyze(N=80, M=10)
