"""Background removal utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

from PIL import Image

ImageInput = Union[str, Path, Image.Image]


def _get_rembg_remove() -> Callable[[Image.Image], Image.Image]:
    """Load rembg lazily to keep import-time dependencies minimal."""
    try:
        from rembg import remove
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "`rembg` 패키지를 찾을 수 없습니다. `pip install rembg`로 설치하세요."
        ) from exc
    return remove


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


def remove_background(image: ImageInput) -> Image.Image:
    """Remove the input image background and return an RGB image on white background."""
    input_image = _load_image(image)
    output_image = _get_rembg_remove()(input_image)

    if output_image.mode != "RGBA":
        output_image = output_image.convert("RGBA")

    white_bg = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
    white_bg.paste(output_image, mask=output_image.split()[3])
    return white_bg.convert("RGB")
