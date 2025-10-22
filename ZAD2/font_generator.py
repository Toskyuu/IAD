import sys
import os
import argparse
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2


def generate_font_image(width, height, x, y, font_file, letter, noise_level, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Utworzono katalog: {output_dir}")


    image_pil = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(image_pil)

    try:
        font_size = int(height * 0.9)
        font = ImageFont.truetype(font_file, font_size)
    except IOError:
        print(f"BŁĄD: Nie można wczytać pliku czcionki: {font_file}")
        sys.exit(1)

    draw.text((x, y), letter, font=font, fill=0)

    image_np = np.array(image_pil)
    _, image_binary = cv2.threshold(image_np, 254, 255, cv2.THRESH_BINARY)
    if noise_level > 0:
        probability = noise_level / 100.0
        for i in range(height):
            for j in range(width):
                if np.random.rand() < probability:
                    image_binary[i][j] = 255 - image_binary[i][j]


    output_image_path = os.path.join(output_dir, f"{letter}.png")
    cv2.imwrite(output_image_path, image_binary)
    print(f"Zapisano obraz: {output_image_path}")

    description_file_path = os.path.join(output_dir, "description.txt")

    line_to_write = f"{letter}.png:letter {letter}, noise level {noise_level}%\n"

    with open(description_file_path, 'a', encoding='utf-8') as f:
        f.write(line_to_write)
    print(f"Zaktualizowano plik: {description_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator obrazów liter do sieci MADALINE.")
    parser.add_argument("w", type=int, help="Szerokość generowanego obrazu (px)")
    parser.add_argument("h", type=int, help="Wysokość generowanego obrazu (px)")
    parser.add_argument("x", type=int, help="Pozycja X litery na obrazie")
    parser.add_argument("y", type=int, help="Pozycja Y litery na obrazie")
    parser.add_argument("font_file", type=str, help="Nazwa pliku .ttf z krojem czcionki")
    parser.add_argument("letter", type=str, help="Pojedyncza litera do wygenerowania")
    parser.add_argument("noise_level", type=int, help="Poziom zaszumienia w procentach (0-100)")
    parser.add_argument("output_directory", type=str, help="Katalog docelowy dla wygenerowanych plików")

    args = parser.parse_args()

    if len(args.letter) != 1:
        print("BŁĄD: Argument 'letter' musi być pojedynczym znakiem.")
        sys.exit(1)

    generate_font_image(
        width=args.w,
        height=args.h,
        x=args.x,
        y=args.y,
        font_file=args.font_file,
        letter=args.letter,
        noise_level=args.noise_level,
        output_dir=args.output_directory
    )