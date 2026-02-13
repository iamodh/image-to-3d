from PIL import Image

from src import bg_remover


def test_remove_background_returns_rgb(monkeypatch):
    input_image = Image.new("RGB", (4, 4), (10, 20, 30))

    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgba.putpixel((1, 1), (255, 0, 0, 255))

    monkeypatch.setattr(bg_remover, "_get_rembg_remove", lambda: (lambda _: rgba))

    result = bg_remover.remove_background(input_image)

    assert result.mode == "RGB"
    assert result.size == (4, 4)
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((1, 1)) == (255, 0, 0)
