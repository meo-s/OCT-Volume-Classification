from __future__ import absolute_import

from typing import Tuple

import PIL.Image
from PIL.Image import Image as PILImage


def load(file_path: str) -> Tuple[PILImage]:
    img = PIL.Image.open(file_path)
    w, h = img.size
    return tuple((img.crop((h * i, 0, h * (i + 1), h))
                  for i in range(w // h)))
