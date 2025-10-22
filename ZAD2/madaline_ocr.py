import sys
import os
import argparse
import cv2
import numpy as np


def load_data_from_directory(directory_path):
    patterns = []
    description_file = os.path.join(directory_path, "description.txt")

    if not os.path.exists(description_file):
        print(f"BŁĄD: Plik description.txt nie został znaleziony w katalogu: {directory_path}")
        sys.exit(1)

    with open(description_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                filename, label = line.split(':', 1)
                label = label.strip()
                image_path = os.path.join(directory_path, filename)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"OSTRZEŻENIE: Nie udało się wczytać obrazu: {image_path}")
                    continue

                image_vector = image.flatten()


                bipolar_vector = np.float64(image_vector)
                bipolar_vector = (bipolar_vector / -127.5) + 1.0

                patterns.append({'label': label, 'vector': bipolar_vector})

            except ValueError:
                print(f"OSTRZEŻENIE: Pomijam niepoprawną linię w description.txt: {line}")


    return patterns


def main(train_dir, test_dir):
    print("Wczytywanie wzorców treningowych...")
    training_patterns = load_data_from_directory(train_dir)
    print("Wczytywanie wzorców testowych...")
    test_patterns = load_data_from_directory(test_dir)

    if not training_patterns:
        print("BŁĄD: Brak poprawnych wzorców treningowych. Zakończono działanie.")
        return
    if not test_patterns:
        print("BŁĄD: Brak poprawnych wzorców testowych. Zakończono działanie.")
        return


    neurons = []
    print("Budowanie sieci MADALINE...")
    for pattern in training_patterns:
        vector = pattern['vector']
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue

        normalized_weights = vector / norm

        neurons.append({
            'label': pattern['label'],
            'weights': normalized_weights
        })

    print(f"Zbudowano sieć z {len(neurons)} neuronami.")
    print("-" * 30)

    for test_pattern in test_patterns:
        test_vector = test_pattern['vector']

        norm = np.linalg.norm(test_vector)
        if norm == 0:
            continue

        normalized_test_vector = test_vector / norm

        best_confidence = -2.0
        recognized_label = "nierozpoznany"

        for neuron in neurons:
            confidence = np.dot(normalized_test_vector, neuron['weights'])
            if confidence > best_confidence:
                best_confidence = confidence
                recognized_label = neuron['label']

        print(f"{test_pattern['label']} -> {recognized_label}, confidence: {best_confidence:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rozpoznawanie znaków OCR za pomocą sieci MADALINE.")
    parser.add_argument("train_directory", type=str, help="Katalog z wzorcami treningowymi")
    parser.add_argument("test_directory", type=str, help="Katalog z wzorcami testowymi")

    args = parser.parse_args()

    main(args.train_directory, args.test_directory)