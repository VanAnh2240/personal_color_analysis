# File: visualize_palettes.py

from PIL import Image, ImageDraw
from classification.palettes import SPRING, SUMMER, AUTUMN, WINTER

def create_palette_image(
    colors,
    output_path,
    block_width=80,
    height=80,
    margin=10,
    bg_color=(240, 240, 240)
):
    """
    Vẽ palette dạng thanh ngang.
    """

    width = len(colors) * block_width + 2 * margin

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for i, color in enumerate(colors):
        x0 = margin + i * block_width
        x1 = x0 + block_width
        draw.rectangle([x0, 0, x1, height], fill=color)

    img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":

    palettes = [
        SPRING,
        SUMMER,
        AUTUMN,
        WINTER,
    ]

    for palette in palettes:
        create_palette_image(
            palette.colors,
            f"{palette.name.lower()}_palette.png"
        )